# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
import torch.nn as nn
import random
import numpy as np
from modules.encoder import EncoderCNN, EncoderLabels
from modules.transformer_decoder import DecoderTransformer
from modules.multihead_attention import MultiheadAttention
from metrics import softIoU, MaskedCrossEntropyCriterion
import pickle
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def label2onehot(labels, pad_value):
    inp_ = torch.unsqueeze(labels, 2)
    tensor = torch.FloatTensor(labels.size(0), labels.size(1), pad_value + 1).zero_().to(device)
    tensor.scatter_(2, inp_, 1)
    tensor, _ = tensor.max(dim=1)
    tensor = tensor[:, :-1]
    tensor[:, 0] = 0
    return tensor


def mask_from_eos(ids, eos_value, mult_before=True):
    mask = torch.ones(ids.size()).to(device).byte()
    mask_aux = torch.ones(ids.size(0)).to(device).byte()

    # find eos in ingredient prediction
    for idx in range(ids.size(1)):
        # force mask to have 1s in the first position to avoid division by 0 when predictions start with eos
        if idx == 0:
            continue
        if mult_before:
            mask[:, idx] = mask[:, idx] * mask_aux
            mask_aux = mask_aux * (ids[:, idx] != eos_value)
        else:
            mask_aux = mask_aux * (ids[:, idx] != eos_value)
            mask[:, idx] = mask[:, idx] * mask_aux
    return mask


def get_model(args, ingr_vocab_size, instrs_vocab_size):
    encoder_ingrs = EncoderLabels(args['embed_size'], ingr_vocab_size,
                                  args['dropout_encoder'], scale_grad=False).to(device)
    encoder_image = EncoderCNN(
        args['embed_size'],
        args['dropout_encoder'],
        args['image_model']
    )
    decoder = DecoderTransformer(
        args['embed_size'],
        instrs_vocab_size,
        dropout=args['dropout_decoder_r'],
        seq_length=args['maxseqlen'],
        num_instrs=args['maxnuminstrs'],
        attention_nheads=args['n_att'], num_layers=args['transf_layers'],
        normalize_before=True,
        normalize_inputs=False,
        last_ln=False,
        scale_embed_grad=False
    )
    ingr_decoder = DecoderTransformer(
        args['embed_size'],
        ingr_vocab_size,
        dropout=args['dropout_decoder_i'],
                                      seq_length=args['maxnumlabels'],
                                      num_instrs=1, attention_nheads=args['n_att_ingrs'],
                                      pos_embeddings=False,
                                      num_layers=args['transf_layers_ingrs'],
                                      learned=False,
                                      normalize_before=True,
                                      normalize_inputs=True,
                                      last_ln=True,
                                      scale_embed_grad=False)
    criterion = MaskedCrossEntropyCriterion(ignore_index=[instrs_vocab_size-1], reduce=False)
    label_loss = nn.BCELoss(reduce=False)
    eos_loss = nn.BCELoss(reduce=False)
    model = InverseCookingModel(encoder_ingrs, decoder, ingr_decoder, encoder_image,
                                crit=criterion, crit_ingr=label_loss, crit_eos=eos_loss,
                                pad_value=ingr_vocab_size-1,
                                ingrs_only=args['ingrs_only'], recipe_only=args['recipe_only'],
                                label_smoothing=args['label_smoothing_ingr'])

    return model


class InverseCookingModel(nn.Module):
    def __init__(self, ingredient_encoder, recipe_decoder, ingr_decoder, image_encoder,
                 crit=None, crit_ingr=None, crit_eos=None,
                 pad_value=0, ingrs_only=True,
                 recipe_only=False, label_smoothing=0.0):

        super(InverseCookingModel, self).__init__()
        self.ingredient_encoder = ingredient_encoder
        self.recipe_decoder = recipe_decoder
        self.image_encoder = image_encoder
        self.ingredient_decoder = ingr_decoder
        self.crit = crit
        self.crit_ingr = crit_ingr
        self.pad_value = pad_value
        self.ingrs_only = ingrs_only
        self.recipe_only = recipe_only
        self.crit_eos = crit_eos
        self.label_smoothing = label_smoothing

    def forward(self, img_inputs, captions, target_ingrs,
                sample=False, keep_cnn_gradients=False):

        if sample:
            return self.sample(img_inputs, greedy=True)

        targets = captions[:, 1:]
        targets = targets.contiguous().view(-1)

        img_features = self.image_encoder(img_inputs, keep_cnn_gradients)

        losses = {}
        target_one_hot = label2onehot(target_ingrs, self.pad_value)
        target_one_hot_smooth = label2onehot(target_ingrs, self.pad_value)

        if not self.recipe_only:
            target_one_hot_smooth[target_one_hot_smooth == 1] = (1-self.label_smoothing)
            target_one_hot_smooth[target_one_hot_smooth == 0] = self.label_smoothing / target_one_hot_smooth.size(-1)
            ingr_ids, ingr_logits = self.ingredient_decoder.sample(None, None, greedy=True,
                                                                   temperature=1.0, img_features=img_features,
                                                                   first_token_value=0, replacement=False)

            ingr_logits = torch.nn.functional.softmax(ingr_logits, dim=-1)
            eos = ingr_logits[:, :, 0]
            target_eos = ((target_ingrs == 0) ^ (target_ingrs == self.pad_value))

            eos_pos = (target_ingrs == 0)
            eos_head = ((target_ingrs != self.pad_value) & (target_ingrs != 0))
            mask_perminv = mask_from_eos(target_ingrs, eos_value=0, mult_before=False)
            ingr_probs = ingr_logits * mask_perminv.float().unsqueeze(-1)

            ingr_probs, _ = torch.max(ingr_probs, dim=1)
            ingr_ids[mask_perminv == 0] = self.pad_value

            ingr_loss = self.crit_ingr(ingr_probs, target_one_hot_smooth)
            ingr_loss = torch.mean(ingr_loss, dim=-1)

            losses['ingr_loss'] = ingr_loss
            losses['card_penalty'] = torch.abs((ingr_probs*target_one_hot).sum(1) - target_one_hot.sum(1)) + \
                                     torch.abs((ingr_probs*(1-target_one_hot)).sum(1))

            eos_loss = self.crit_eos(eos, target_eos.float())

            mult = 1/2
            losses['eos_loss'] = mult*(eos_loss * eos_pos.float()).sum(1) / (eos_pos.float().sum(1) + 1e-6) + \
                                 mult*(eos_loss * eos_head.float()).sum(1) / (eos_head.float().sum(1) + 1e-6)
            pred_one_hot = label2onehot(ingr_ids, self.pad_value)
            losses['iou'] = softIoU(pred_one_hot, target_one_hot)

        if self.ingrs_only:
            return losses

        target_ingr_feats = self.ingredient_encoder(target_ingrs)
        target_ingr_mask = mask_from_eos(target_ingrs, eos_value=0, mult_before=False)

        target_ingr_mask = target_ingr_mask.float().unsqueeze(1)

        outputs, ids = self.recipe_decoder(target_ingr_feats, target_ingr_mask, captions, img_features)

        outputs = outputs[:, :-1, :].contiguous()
        outputs = outputs.view(outputs.size(0) * outputs.size(1), -1)
        loss = self.crit(outputs, targets)

        losses['recipe_loss'] = loss
        return losses

    def sample(self, img_inputs, greedy=True, temperature=1.0, beam=-1, true_ingrs=None):

        outputs = dict()

        img_features = self.image_encoder(img_inputs)

        if not self.recipe_only:
            ingr_ids, ingr_probs = self.ingredient_decoder.sample(None, None, greedy=True, temperature=temperature,
                                                                  beam=-1,
                                                                  img_features=img_features, first_token_value=0,
                                                                  replacement=False)

            sample_mask = mask_from_eos(ingr_ids, eos_value=0, mult_before=False)
            ingr_ids[sample_mask == 0] = self.pad_value

            outputs['ingr_ids'] = ingr_ids
            outputs['ingr_probs'] = ingr_probs.data

            mask = sample_mask
            input_mask = mask.float().unsqueeze(1)
            input_feats = self.ingredient_encoder(ingr_ids)

        if self.ingrs_only:
            return outputs

        if true_ingrs is not None:
            input_mask = mask_from_eos(true_ingrs, eos_value=0, mult_before=False)
            true_ingrs[input_mask == 0] = self.pad_value
            input_feats = self.ingredient_encoder(true_ingrs)
            input_mask = input_mask.unsqueeze(1)

        ids, probs = self.recipe_decoder.sample(input_feats, input_mask, greedy, temperature, beam, img_features, 0,
                                                last_token_value=1)

        outputs['recipe_probs'] = probs.data
        outputs['recipe_ids'] = ids

        return outputs
