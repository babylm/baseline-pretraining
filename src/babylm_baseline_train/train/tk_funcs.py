import os
import pdb
import setuptools
import torch

from transformers import AutoTokenizer
from transformers import GPT2Tokenizer


def get_gpt2_tokenizer_func(model_name='gpt2'):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    return tokenizer


def get_roberta_tokenizer_func(model_name="roberta-base"):
    from transformers import RobertaTokenizer
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    return tokenizer


def get_tokenizer_func(opt_model_size='125m'):
    model_name = f"facebook/opt-{opt_model_size}"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.add_bos_token = False
    tokenizer.add_special_tokens(
            {
                'bos_token': '<s>', 
                'unk_token': '<unk>',
                'additional_special_tokens': [
                    '<image>', '</c>', 
                    '<PERSON>', # C-12M for person names
                    ]
            })
    return tokenizer
