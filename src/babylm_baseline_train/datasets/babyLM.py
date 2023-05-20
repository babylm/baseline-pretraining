from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
import numpy as np
import pdb

from .utils import Group_Texts
from .base import BaseGroupDataset
from . import hf_loaders


class BabyLM(BaseGroupDataset):
    def __init__(
            self, 
            seq_len=128, tokenizer=None,
            name='babyLM-10M',
            ):
        super().__init__(seq_len, tokenizer)
        self.name = name

    def get_dataset(self):
        self.dataset = hf_loaders.get_babyLM(
                name=self.name,
                split="train")


def get_babyLM_10M(seq_len=128, tokenizer=None, just_dataset=False):
    dataset_builder = BabyLM(
            seq_len=seq_len,
            tokenizer=tokenizer,
            name='babyLM-10M',
            )
    return dataset_builder.get_group_dataset(just_dataset=just_dataset)


def get_babyLM_100M(seq_len=128, tokenizer=None, just_dataset=False):
    dataset_builder = BabyLM(
            seq_len=seq_len,
            tokenizer=tokenizer,
            name='babyLM-100M',
            )
    return dataset_builder.get_group_dataset(just_dataset=just_dataset)
