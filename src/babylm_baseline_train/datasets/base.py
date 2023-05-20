from transformers import AutoTokenizer
import torch
from abc import ABC, abstractmethod
from transformers import GPT2Tokenizer
import ipdb
from tqdm import tqdm

from .utils import Group_Texts


class BaseGroupDataset(ABC):
    def __init__(self, seq_len, tokenizer):
        self.seq_len = seq_len
        self.tokenizer = tokenizer

    def prepare_tokenizer(self):
        if self.tokenizer is None:
            #self.tokenizer = AutoTokenizer.from_pretrained(
            #        "gpt2", fast=False)
            model_name = f"facebook/opt-125m"
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    @abstractmethod
    def get_dataset(self):
        pass

    def tokenize_function(self, examples):
        outputs = self.tokenizer(examples['text'])
        return outputs

    def get_group_dataset(self, just_dataset=False):
        self.prepare_tokenizer()
        self.get_dataset()
        if just_dataset == True:
            return self.dataset
        elif just_dataset == 'self':
            return self

        tokenized_datasets = self.dataset.map(
                self.tokenize_function, batched=True, 
                remove_columns=["text"])
        group_text_default = Group_Texts(
                tokenized_datasets, self.tokenizer, 
                seq_len=self.seq_len)

        grouped_dataset_default = group_text_default.group_texts()
        return grouped_dataset_default

    def count_num_of_words(self):
        import re
        import inflect
        import nltk.data
        from tqdm import tqdm

        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        num = 0
        for data in tqdm(self.dataset):
            sents = tokenizer.tokenize(data['text'])
            for sent in sents:
                tokens = re.findall('\w+', sent)
                num += len(tokens)
        return num

    def count_num_of_tks(self):
        num_of_tks = 0
        for line in tqdm(self.dataset):
            txt_in_tks = self.tokenize_function(line)
            num_of_tks += len(txt_in_tks.input_ids)
        return num_of_tks
