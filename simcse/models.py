import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import copy
import transformers
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertLMPredictionHead, BertModel
from transformers.models.xlm_roberta.configuration_xlm_roberta import XLMRobertaConfig
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaModel
from transformers.activations import gelu
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions

#from .modules import MyBertModel as BertModel

class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x

class ozhang_MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class azhang_MLP(nn.Module):
    "weak"
    def __init__(self, in_dim, out_dim, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class bzhang_MLP(nn.Module):
    "not stable"
    def __init__(self, in_dim, out_dim, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class czhang_MLP(nn.Module):
    "This one is strong"
    def __init__(self, in_dim, out_dim, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        return self.net(x)

class dzhang_MLP(nn.Module):
    "not deep enough"
    def __init__(self, in_dim, out_dim, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        return self.net(x)

class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls', 'simcse_sup']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def build_mlp(mlp_type, cls, config):
    if mlp_type == 'identity':
        mlp = nn.Identity()
    elif mlp_type[1:].startswith('zhang'):
        nblocks = int(mlp_type[6:].split('-')[0])
        hidden_multiplier = int(mlp_type[6:].split('-')[1])
        mlp_factory = {'o': ozhang_MLP, 'a': azhang_MLP, 'b': bzhang_MLP, 'c': czhang_MLP, 'd': dzhang_MLP}
        project_list = [mlp_factory[mlp_type[0]](config.hidden_size, config.hidden_size, hidden_multiplier * config.hidden_size) for i in range(nblocks)]
        mlp = nn.Sequential(*project_list)
    elif mlp_type == 'simcse_sup':
        mlp = MLPLayer(config)
    else:
        assert False, f"{mlp_type} not supported"
    return mlp

def cl_init(cls, encoder, config):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)
    cls.mlp = build_mlp(cls.model_args.proj_mlp_type, cls, config)

    if cls.model_args.training_method in ['simsiam', 'byol']:
        cls.mlp_pred = build_mlp(cls.model_args.pred_mlp_type, cls, config)

    if cls.model_args.training_method == 'barlow':
        cls.bn = nn.BatchNorm1d(config.hidden_size, affine=False)
    cls.sim = nn.CosineSimilarity(dim=-1)
    cls.init_weights()
    if cls.model_args.training_method == 'byol':
        cls.target_encoder = copy.deepcopy(encoder)
        cls.target_mlp = copy.deepcopy(cls.mlp)

def cl_forward(cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    mlm_input_ids=None,
    mlm_labels=None,
#    is_amr=None,
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    ori_input_ids = input_ids
    batch_size = input_ids.size(0)
    # Number of sentences in one instance
    # 2: pair instance; 3: pair instance with a hard negative
    num_sent = input_ids.size(1)

    mlm_outputs = None
    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
    attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)
#    if is_amr is not None:
#        is_amr = is_amr.view(-1)
    # Get raw embeddings
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
#        is_amr=is_amr,
    )

    # MLM auxiliary objective
    if mlm_input_ids is not None:
        mlm_input_ids = mlm_input_ids.view((-1, mlm_input_ids.size(-1)))
        mlm_outputs = encoder(
            mlm_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
        )

    # Pooling
    pooler_output = cls.pooler(attention_mask, outputs)
    pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1))) # (bs, num_sent, hidden)

    # projector
    # Separate representation
    z1, z2 = cls.mlp(pooler_output[:,0]), cls.mlp(pooler_output[:,1])

    # Hard negative
    if num_sent == 3:
        z3 = cls.mlp(pooler_output[:, 2])

    # Gather all embeddings if using distributed training
    if dist.is_initialized() and cls.training:
        # Gather hard negative
        if num_sent >= 3:
            z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
            z3_list[dist.get_rank()] = z3
            z3 = torch.cat(z3_list, 0)

        # Dummy vectors for allgather
        z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
        z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
        # Allgather
        dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
        dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

        # Since allgather results do not have gradients, we replace the
        # current process's corresponding embeddings with original tensors
        z1_list[dist.get_rank()] = z1
        z2_list[dist.get_rank()] = z2
        # Get full batch embeddings: (bs x N, hidden)
        z1 = torch.cat(z1_list, 0)
        z2 = torch.cat(z2_list, 0)
    N, D = z1.size()
    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0)) / cls.model_args.temp
    if cls.model_args.training_method == "simcse":
        # negative
        # Hard negative
        if num_sent == 3:
            z1_z3_cos = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0)) / cls.model_args.temp
            cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)
            # Calculate loss with hard negatives
            # Note that weights are actually logits of weights
            z3_weight = cls.model_args.hard_negative_weight
            weights = torch.tensor(
                [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
            ).to(cls.device)
            cos_sim = cos_sim + weights

        labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
        loss_fct = nn.CrossEntropyLoss() 
        loss = loss_fct(cos_sim, labels) + loss_fct(cos_sim[:N,:N].T, labels)
    elif cls.model_args.training_method == "simsiam":
        p1, p2 = cls.mlp_pred(z1), cls.mlp_pred(z2)
        loss = -(cls.sim(p1, z2.detach()) + cls.sim(p2, z1.detach()))
        loss = loss.mean()
    elif cls.model_args.training_method == "byol":
        # target network, pred_mlp,
        # (+barlow) (+negative)
        with torch.no_grad():
            outputs_target = cls.target_encoder(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
                    return_dict=True,
#                    is_amr=is_amr,
            )

            # Pooling
            pooler_output_target = cls.pooler(attention_mask, outputs_target)
            pooler_output_target = pooler_output_target.view((batch_size, num_sent, pooler_output_target.size(-1))) # (bs, num_sent, hidden)

            # Projector
            # Separate representation
            z1_target, z2_target = cls.target_mlp(pooler_output_target[:,0]), cls.target_mlp(pooler_output_target[:,1])
            # Gather all embeddings if using distributed training
            if dist.is_initialized() and cls.training:
                # Dummy vectors for allgather
                z1_list = [torch.zeros_like(z1_target) for _ in range(dist.get_world_size())]
                z2_list = [torch.zeros_like(z2_target) for _ in range(dist.get_world_size())]
                # Allgather
                dist.all_gather(tensor_list=z1_list, tensor=z1_target.contiguous())
                dist.all_gather(tensor_list=z2_list, tensor=z2_target.contiguous())

                # Since allgather results do not have gradients, we replace the
                # current process's corresponding embeddings with original tensors
                z1_list[dist.get_rank()] = z1_target
                z2_list[dist.get_rank()] = z2_target
                # Get full batch embeddings: (bs x N, hidden)
                z1_target = torch.cat(z1_list, 0)
                z2_target = torch.cat(z2_list, 0)
        p1, p2 = cls.mlp_pred(z1), cls.mlp_pred(z2)
        loss = -(cls.sim(p1, z2_target.detach()) + cls.sim(p2, z1_target.detach()))
        loss = loss.mean()
    elif cls.model_args.training_method == "barlow":
        z1_norm = cls.bn(z1)
        z2_norm = cls.bn(z2)
        c = torch.mm(z1_norm.T, z2_norm) / N
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + cls.model_args.lambd * off_diag
    elif cls.model_args.training_method == "vicreg":
        sim_loss = F.mse_loss(z1, z2)

        eps = 1e-4
        std_z1 = torch.sqrt(z1.var(dim=0) + eps)
        std_z2 = torch.sqrt(z2.var(dim=0) + eps)
        var_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))

        z1_norm = z1 - z1.mean(dim=0)
        z2_norm = z2 - z2.mean(dim=0)
        cov_z1 = (z1_norm.T @ z1_norm) / (N - 1)
        cov_z2 = (z2_norm.T @ z2_norm) / (N - 1)
        cov_loss = off_diagonal(cov_z1).pow_(2).sum() / D + off_diagonal(cov_z2).pow_(2).sum() / D
        
        loss = cls.model_args.sim_loss_weight * sim_loss + \
               cls.model_args.var_loss_weight * var_loss + \
               cls.model_args.cov_loss_weight * cov_loss
    else:
        assert False

    # Calculate loss for MLM
    if mlm_outputs is not None and mlm_labels is not None:
        mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
        prediction_scores = cls.lm_head(mlm_outputs.last_hidden_state)
        vocab_size = prediction_scores.size(-1)
        masked_lm_loss = loss_fct(prediction_scores.view(-1, vocab_size), mlm_labels.view(-1))
        loss = loss + cls.model_args.mlm_weight * masked_lm_loss

    if not return_dict:
        output = (cos_sim,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
    return SequenceClassifierOutput(
        loss=loss,
        logits=cos_sim,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def sentemb_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
#    is_amr=None,
):

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
#        is_amr=is_amr,
    )

    pooler_output = cls.pooler(attention_mask, outputs)
    if cls.model_args.after_projector:
        pooler_output = cls.mlp(pooler_output)

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )


class BertForCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.bert = BertModel(config, add_pooling_layer=False)

        if self.model_args.do_mlm:
            self.lm_head = BertLMPredictionHead(config)

        cl_init(self, self.bert, config)
    
    def get_output_embeddings(self):
        if self.model_args.do_mlm:
            return self.lm_head.decoder
        else:
            return None

    def update_moving_average(self, replace=False):
        if self.model_args.training_method != "byol":
            return False

        for current_params, ma_params in zip(self.bert.parameters(), self.target_encoder.parameters()):
            old, new = ma_params.data, current_params.data
            ma_params.data = new if replace else old * self.model_args.beta + (1 - self.model_args.beta) * new
        for current_params, ma_params in zip(self.mlp.parameters(), self.target_mlp.parameters()):
            old, new = ma_params.data, current_params.data
            ma_params.data = new if replace else old * self.model_args.beta + (1 - self.model_args.beta) * new
        return True

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
#        is_amr=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
#                is_amr=is_amr,
            )
        else:
            return cl_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
#                is_amr=is_amr,
            )



class RobertaForCL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.roberta = RobertaModel(config, add_pooling_layer=False)

        if self.model_args.do_mlm:
            self.lm_head = RobertaLMHead(config)

        cl_init(self, self.roberta, config)
    
    def get_output_embeddings(self):
        if self.model_args.do_mlm:
            return self.lm_head.decoder
        else:
            return None

    def update_moving_average(self, replace=False):
        if self.model_args.training_method != "byol":
            return False

        for current_params, ma_params in zip(self.roberta.parameters(), self.target_encoder.parameters()):
            old, new = ma_params.data, current_params.data
            ma_params.data = new if replace else old * self.model_args.beta + (1 - self.model_args.beta) * new
        for current_params, ma_params in zip(self.mlp.parameters(), self.target_mlp.parameters()):
            old, new = ma_params.data, current_params.data
            ma_params.data = new if replace else old * self.model_args.beta + (1 - self.model_args.beta) * new
        return True
    
    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )

class XLMRobertaForCL(RobertaForCL):
    config_class = XLMRobertaConfig
    
    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)

        if self.model_args.do_mlm:
            self.lm_head = RobertaLMHead(config)

        cl_init(self, self.roberta, config)
