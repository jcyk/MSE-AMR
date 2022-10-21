# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
SST - binary classification
'''

from __future__ import absolute_import, division, unicode_literals

import os
import io
import logging
import numpy as np
import json
import random
from senteval.tools.validation import SplitClassifier


class MARCEval(object):
    def __init__(self, task_path, seed=1111, zero_shot=False):
        self.seed = seed
        self.zero_shot = zero_shot
                
        self.nclasses = 5
        logging.debug('***** Transfer task : MARC classification (zero-shot = {}) *****\n\n'.format(zero_shot))
        
        self.sst_data = {}
        self.langs = ['de','en','es','fr','ja','zh']
        
        if not zero_shot:
            for lang in self.langs:
                self.sst_data['train.{}'.format(lang)] = self.loadFile(os.path.join(task_path, 'train.{}.json'.format(lang)))
        else:
            self.sst_data['train.en'] = self.loadFile(os.path.join(task_path, 'train.en.json'))
        for lang in self.langs:
            self.sst_data['dev.{}'.format(lang)] = self.loadFile(os.path.join(task_path, 'dev.{}.json'.format(lang)))
            self.sst_data['test.{}'.format(lang)] = self.loadFile(os.path.join(task_path, 'test.{}.json'.format(lang)))

    def do_prepare(self, params, prepare):
#        samples = self.sst_data['train']['X'] + self.sst_data['dev']['X'] + \
#                  self.sst_data['test']['X']
#        return prepare(params, samples)
        return

    def loadFile(self, fpath):
        sst_data = {'X': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                j = json.loads(line.rstrip())
                sst_data['y'].append(int(j['stars'])-1)
                sst_data['X'].append(j['review_body'].split())
        assert max(sst_data['y']) == self.nclasses - 1
        return sst_data

    def run(self, params, batcher):
        train_langs = self.langs if not self.zero_shot else ['en']
        sst_embed = {}
        for lang in train_langs:
            sst_embed['train.{}'.format(lang)] = {}
        for lang in self.langs:
            sst_embed['dev.{}'.format(lang)] = {}
            sst_embed['test.{}'.format(lang)] = {}
        bsize = params.batch_size

        for key in self.sst_data:
            logging.info('Computing embedding for {0}'.format(key))
            # Sort to reduce padding
            sorted_data = sorted(zip(self.sst_data[key]['X'],
                                     self.sst_data[key]['y']),
                                 key=lambda z: (len(z[0]), z[1]))
            self.sst_data[key]['X'], self.sst_data[key]['y'] = map(list, zip(*sorted_data))

            sst_embed[key]['X'] = []
            for ii in range(0, len(self.sst_data[key]['y']), bsize):
                batch = self.sst_data[key]['X'][ii:ii + bsize]
                embeddings = batcher(params, batch, lang=key[-2:])
                sst_embed[key]['X'].append(embeddings)
            sst_embed[key]['X'] = np.vstack(sst_embed[key]['X'])
            sst_embed[key]['y'] = np.array(self.sst_data[key]['y'])
            logging.info('Computed {0} embeddings'.format(key))

        config_classifier = {'nclasses': self.nclasses, 'seed': self.seed,
                             'usepytorch': params.usepytorch,
                             'classifier': params.classifier}

        X_train = [sst_embed['train.{}'.format(lang)]['X'] for lang in train_langs]
        y_train = [sst_embed['train.{}'.format(lang)]['y'] for lang in train_langs]
        X_train = np.vstack(X_train)
        y_train = np.concatenate(y_train)
        
        X={'train': X_train, 'valid': {}, 'test': {}}
        y={'train': y_train, 'valid': {}, 'test': {}}
        for lang in self.langs:
            X['valid'][lang] = sst_embed['dev.{}'.format(lang)]['X']
            y['valid'][lang] = sst_embed['dev.{}'.format(lang)]['y']
            X['test'][lang] = sst_embed['test.{}'.format(lang)]['X']
            y['test'][lang] = sst_embed['test.{}'.format(lang)]['y']
        clf = SplitClassifier(X=X,
                              y=y,
                              config=config_classifier,
                              multi_dev=True)

        devacc, testacc = clf.run()
        logging.debug(('\nDev acc : {0} Test acc : {1} ' 
            'for MARC classification (zero-shot = {2})\n').format(devacc, ', '.join(['({}) {}'.format(k, v) for k, v in testacc.items()]), self.zero_shot))
#        for lang in self.langs:
#            logging.debug(('\nDev acc : {0} Test acc : {1} for {2} for '
#                'MARC classification\n').format(devacc[lang], testacc[lang], lang))

        return {'devacc': devacc, 'acc': testacc}
                #'ndev': len(sst_embed['dev']['X']),
                #'ntest': len(sst_embed['test']['X'])}
