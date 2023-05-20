import os
import pdb
import setuptools
import torch
import ipdb
import copy

from transformers import OPTForCausalLM
from transformers.models.opt.modeling_opt import OPTConfig
DEBUG = int(os.environ.get(
        'DEBUG',
        '0')) == 1


def get_opt_func(opt_model_size='125m'):
    model_name = f"facebook/opt-{opt_model_size}"
    config = OPTConfig.from_pretrained(model_name)
    model = OPTForCausalLM(config=config)
    return model


def get_roberta_func(model_name="roberta-base", tokenizer=None):
    from transformers import RobertaConfig, RobertaForMaskedLM
    config = RobertaConfig.from_pretrained(model_name)
    model = RobertaForMaskedLM(config)
    if tokenizer is not None:
        model.resize_token_embeddings(len(tokenizer))
    return model
