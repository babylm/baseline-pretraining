import os
import ipdb
import setuptools
import torch

import babylm_baseline_train.train.tk_funcs as tk_funcs
import babylm_baseline_train.models.helper as helper


def load_opt125m():
    tokenizer = tk_funcs.get_tokenizer_func()
    model = helper.get_opt_func()
    saved_model = torch.load(
            './babyLM_10M/opt125m_s1/epoch_20.pth', # path to your pretrained model
            map_location=torch.device('cpu'))
    model.load_state_dict(saved_model['state_dict'])


def load_roberta():
    tokenizer = tk_funcs.get_roberta_tokenizer_func()
    model = helper.get_roberta_func(tokenizer=tokenizer)
    saved_model = torch.load(
            './babyLM_10M/roberta_s1/epoch_20.pth', # path to your pretrained model
            map_location=torch.device('cpu'))
    saved_model['state_dict'].pop('roberta.embeddings.token_type_ids')
    model.load_state_dict(saved_model['state_dict'])
