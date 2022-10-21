# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Pytorch Classifier class in the style of scikit-learn
Classifiers include Logistic Regression and MLP
"""

from __future__ import absolute_import, division, unicode_literals

import numpy as np
import copy
from senteval import utils

import torch
from torch import nn
import torch.nn.functional as F
import collections

class PyTorchClassifier(object):
    def __init__(self, inputdim, nclasses, l2reg=0., batch_size=64, seed=1111,
                 cudaEfficient=False, multi_dev=False, combine_dev=True):
        # fix seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.inputdim = inputdim
        self.nclasses = nclasses
        self.l2reg = l2reg
        self.batch_size = batch_size
        self.cudaEfficient = cudaEfficient

        self.multi_dev = multi_dev
        self.combine_dev = combine_dev
        self.models = None
        
    def prepare_split(self, X, y, validation_data=None, validation_split=None):
        # Preparing validation data
        if self.multi_dev:
            assert validation_data
        else:
            assert validation_split or validation_data
        if validation_data is not None:
            trainX, trainy = X, y
            devX, devy = validation_data
        else:
            permutation = np.random.permutation(len(X))
            trainidx = permutation[int(validation_split * len(X)):]
            devidx = permutation[0:int(validation_split * len(X))]
            trainX, trainy = X[trainidx], y[trainidx]
            devX, devy = X[devidx], y[devidx]

        device = torch.device('cpu') if self.cudaEfficient else torch.device('cuda')

        trainX = torch.from_numpy(trainX).to(device, dtype=torch.float32)
        trainy = torch.from_numpy(trainy).to(device, dtype=torch.int64)
        if self.multi_dev:
            devX_ = {}
            devy_ = {}
            for key, value in devX.items():
                devX_[key] = torch.from_numpy(value).to(device, dtype=torch.float32)
            for key, value in devy.items():
                devy_[key] = torch.from_numpy(value).to(device, dtype=torch.int64)
            devX = devX_
            devy = devy_
        else:
            devX = torch.from_numpy(devX).to(device, dtype=torch.float32)
            devy = torch.from_numpy(devy).to(device, dtype=torch.int64)

        return trainX, trainy, devX, devy

    def fit(self, X, y, validation_data=None, validation_split=None,
            early_stop=True):
        self.nepoch = 0
        
        if self.multi_dev and not self.combine_dev:
            assert validation_data
            bestaccuracy = {key: -1 for key in validation_data[0]}
            stop_train = {key: False for key in validation_data[0]}        
            early_stop_count = {key: 0 for key in validation_data[0]}
        else:
            bestaccuracy = -1
            stop_train = False
            early_stop_count = 0

#        if self.multi_dev and self.combine_dev:
#            keys = list(validation_data[1].keys())
#            validation_data = [[data[key] for key in keys] for data in validation_data]
#            if len(validation_data[0]) > 0:
#                validation_data = [np.vstack(d) if d[0].ndim > 1 else np.concatenate(d) for d in validation_data]
            
        # Preparing validation data
        trainX, trainy, devX, devy = self.prepare_split(X, y, validation_data,
                                                        validation_split)

        # Training
        if not self.multi_dev or self.combine_dev:
            while not stop_train and self.nepoch <= self.max_epoch:
                self.trainepoch(trainX, trainy, epoch_size=self.epoch_size)
                if not self.multi_dev:                
                    accuracy = self.score(devX, devy)
                else:
                    accuracy = np.mean([self.score(devX[key], devy[key]) for key in devX])                                    
                if accuracy > bestaccuracy:
                    bestaccuracy = accuracy
                    bestmodel = copy.deepcopy(self.model)
                elif early_stop:
                    if early_stop_count >= self.tenacity:
                        stop_train = True
                    early_stop_count += 1
            self.model = bestmodel                    
        else:
            bestmodel = {}
            while not all(stop_train.values()) and self.nepoch <= self.max_epoch:
                self.trainepoch(trainX, trainy, epoch_size=self.epoch_size)
                
                for key in devX:
                    if stop_train[key]:
                        continue
                    accuracy = self.score(devX[key], devy[key])
                    if accuracy > bestaccuracy[key]:
                        bestaccuracy[key] = accuracy
                        bestmodel[key] = copy.deepcopy(self.model)
                    elif early_stop:
                        if early_stop_count[key] >= self.tenacity:
                            stop_train[key] = True
                        early_stop_count[key] += 1
            self.models = bestmodel
        return bestaccuracy

    def trainepoch(self, X, y, epoch_size=1):
        self.model.train()
        for _ in range(self.nepoch, self.nepoch + epoch_size):
            permutation = np.random.permutation(len(X))
            all_costs = []
            for i in range(0, len(X), self.batch_size):
                # forward
                idx = torch.from_numpy(permutation[i:i + self.batch_size]).long().to(X.device)

                Xbatch = X[idx]
                ybatch = y[idx]

                if self.cudaEfficient:
                    Xbatch = Xbatch.cuda()
                    ybatch = ybatch.cuda()
                output = self.model(Xbatch)
                # loss
                loss = self.loss_fn(output, ybatch)
                all_costs.append(loss.data.item())
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                # Update parameters
                self.optimizer.step()
        self.nepoch += epoch_size

    def score(self, devX, devy, model_key=None):
        if isinstance(devX, collections.Mapping):
            return {key: self.score(devX[key], devy[key], key) for key in devX}
        if self.multi_dev and not self.combine_dev:
            assert model_key is not None and model_key in self.models or \
                model_key is None and self.models is None
            if model_key is not None:
                self.model = self.models[model_key]
        self.model.eval()
        correct = 0
        if not isinstance(devX, torch.cuda.FloatTensor) or self.cudaEfficient:
            devX = torch.FloatTensor(devX).cuda()
            devy = torch.LongTensor(devy).cuda()
        with torch.no_grad():
            for i in range(0, len(devX), self.batch_size):
                Xbatch = devX[i:i + self.batch_size]
                ybatch = devy[i:i + self.batch_size]
                if self.cudaEfficient:
                    Xbatch = Xbatch.cuda()
                    ybatch = ybatch.cuda()
                output = self.model(Xbatch)
                pred = output.data.max(1)[1]
                correct += pred.long().eq(ybatch.data.long()).sum().item()
            accuracy = 1.0 * correct / len(devX)
        return accuracy

    def predict(self, devX):
        if isinstance(devX, collections.Mapping):
            return {key: self.predict(devX[key]) for key in devX}
        self.model.eval()
        if not isinstance(devX, torch.cuda.FloatTensor):
            devX = torch.FloatTensor(devX).cuda()
        yhat = np.array([])
        with torch.no_grad():
            for i in range(0, len(devX), self.batch_size):
                Xbatch = devX[i:i + self.batch_size]
                output = self.model(Xbatch)
                yhat = np.append(yhat,
                                 output.data.max(1)[1].cpu().numpy())
        yhat = np.vstack(yhat)
        return yhat

    def predict_proba(self, devX):
        if isinstance(devX, collections.Mapping):
            return {key: self.predict_proba(devX[key]) for key in devX}
        self.model.eval()
        probas = []
        with torch.no_grad():
            for i in range(0, len(devX), self.batch_size):
                Xbatch = devX[i:i + self.batch_size]
                vals = F.softmax(self.model(Xbatch).data.cpu().numpy())
                if not probas:
                    probas = vals
                else:
                    probas = np.concatenate(probas, vals, axis=0)
        return probas


"""
MLP with Pytorch (nhid=0 --> Logistic Regression)
"""

class MLP(PyTorchClassifier):
    def __init__(self, params, inputdim, nclasses, l2reg=0., batch_size=64,
                 seed=1111, cudaEfficient=False, multi_dev=False):
        super(self.__class__, self).__init__(inputdim, nclasses, l2reg,
                                             batch_size, seed, cudaEfficient, multi_dev)
        """
        PARAMETERS:
        -nhid:       number of hidden units (0: Logistic Regression)
        -optim:      optimizer ("sgd,lr=0.1", "adam", "rmsprop" ..)
        -tenacity:   how many times dev acc does not increase before stopping
        -epoch_size: each epoch corresponds to epoch_size pass on the train set
        -max_epoch:  max number of epoches
        -dropout:    dropout for MLP
        """
        
        self.nhid = 0 if "nhid" not in params else params["nhid"]
        self.optim = "adam" if "optim" not in params else params["optim"]
        self.tenacity = 5 if "tenacity" not in params else params["tenacity"]
        self.epoch_size = 4 if "epoch_size" not in params else params["epoch_size"]
        self.max_epoch = 200 if "max_epoch" not in params else params["max_epoch"]
        self.dropout = 0. if "dropout" not in params else params["dropout"]
        self.batch_size = 64 if "batch_size" not in params else params["batch_size"]

        if params["nhid"] == 0:
            self.model = nn.Sequential(
                nn.Linear(self.inputdim, self.nclasses),
            ).cuda()
        else:
            self.model = nn.Sequential(
                nn.Linear(self.inputdim, params["nhid"]),
                nn.Dropout(p=self.dropout),
                nn.Sigmoid(),
                nn.Linear(params["nhid"], self.nclasses),
            ).cuda()

        self.loss_fn = nn.CrossEntropyLoss().cuda()
        self.loss_fn.size_average = False

        optim_fn, optim_params = utils.get_optimizer(self.optim)
        self.optimizer = optim_fn(self.model.parameters(), **optim_params)
        self.optimizer.param_groups[0]['weight_decay'] = self.l2reg
