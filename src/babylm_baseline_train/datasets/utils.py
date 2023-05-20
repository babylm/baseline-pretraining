from copy import deepcopy
from typing import Optional
import torch


# New class based off of group_texts with stride, padding, and padding token inputs
class Group_Texts:
    def __init__(self,
                 tokenized_dataset,
                 tokenizer,
                 seq_len: int,
                 stride: Optional[int] = None,
                 padding: Optional[bool] = False,
                 padding_tok: Optional[int] = None
                 ):
        # Set values for the class variables
        self.dataset = tokenized_dataset
        self.seq_len = seq_len

        # if-else for setting stride/padding/padding token
        # Padding false, stride None -> Default
        if padding is False and stride is None:
            self.stride = seq_len
            self.padding = padding
        # Padding true, stride None -> Only padding
        elif padding is True and stride is None:
            self.stride = seq_len
            self.padding = padding
            if padding_tok is not None:
                self.padding_tok = padding_tok
            elif padding_tok is None:
                # Doesn't matter what the padding token is since it will be masked dually by labels and attention mask
                # Can also set to the input id value of eos token
                self.padding_tok = (tokenizer(tokenizer.eos_token))["input_ids"][0]
                print(
                    f'Padding token defaulting to {(tokenizer(tokenizer.eos_token))["input_ids"][0]} (debugging), it will be masked by labels and attention mask')
        # Padding false, stride a value -> Only stride
        elif padding is False and stride is not None:
            self.stride = stride
            self.padding = padding
        # Padding true, stride a value -> Stride with padding
        elif padding is True and stride is not None:
            self.stride = stride
            self.padding = padding
            if padding_tok is not None:
                self.padding_tok = padding_tok
            elif padding_tok is None:
                self.padding_tok = (tokenizer(tokenizer.eos_token))["input_ids"][0]
                print(
                    f'Padding token defaulting to {(tokenizer(tokenizer.eos_token))["input_ids"][0]} (debugging), it will be masked by labels and attention mask')

        # Split function calls by the inputs
        if self.padding is False and self.stride is self.seq_len:
            print("Grouping texts with default mode without padding or stride at context length of", self.seq_len)
        elif self.padding is True and self.stride is self.seq_len:
            print("Grouping texts with padding with padding token", self.padding_tok, "at context length of", self.seq_len)
        elif self.padding is False and self.stride is not self.seq_len:
            print("Grouping texts at a stride of", self.stride, "at context length of", self.seq_len)
        elif self.padding is True and self.stride is not self.seq_len:
            print("Grouping texts with padding with padding token", self.padding_tok, "and stride of", self.stride, "at context length of", self.seq_len)

    def group_texts(self):
        # Call preferred grouping function
        return self.dataset.map(self.get_grouping, batched=True, batch_size=1000)

    # Default function with no padding or striding
    # Leaves out tokens that do not fit into a multiple of seq_len
    def group_default(self, examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length_use = (total_length // self.seq_len) * self.seq_len
        result = {
            k: [t[i: i + self.seq_len] for i in range(0, total_length_use, self.seq_len)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()

        # Some checks
        assert all([len(x) == self.seq_len for x in result["input_ids"]])
        if 'attention_mask' in result:
            assert all([len(x) == self.seq_len for x in result["attention_mask"]])
        assert all([len(x) == self.seq_len for x in result["labels"]])

        return result

    # Only Padding function
    # Takes the left out tokens and pads to seq_len
    def group_padding(self, examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        # Find what length to add padding
        remainder = total_length % self.seq_len
        if remainder != 0:
            to_add = self.seq_len - remainder
        elif remainder == 0:
            to_add = 0
        to_add_input_id = [self.padding_tok] * to_add
        to_add_atten_mask = [0] * to_add
        # Merge the two Dict variables
        pad_dict = dict(input_ids=to_add_input_id, attention_mask=to_add_atten_mask)
        for key in concatenated_examples.keys():
            t = concatenated_examples[key]
            t1 = [item for sublist in [t, pad_dict[key]] for item in sublist]
            assert not len(t1) % self.seq_len
            concatenated_examples[key] = t1
        total_length_use = len(concatenated_examples[list(examples.keys())[0]])
        result = {
            k: [t[i: i + self.seq_len] for i in range(0, total_length_use, self.seq_len)]
            for k, t in concatenated_examples.items()
        }
        # Labels is copied from input ids
        result["labels"] = result["input_ids"].copy()

        # Label is -100 if attention mask is 0, otherwise same as input ids
        result["labels"] = [
            [-100 if mask == 0 else token for mask, token in mask_and_tokens] for mask_and_tokens in
            [zip(masks, labels) for masks, labels in zip(result["attention_mask"], result["labels"])]
        ]

        # Some checks
        assert all([len(x) == self.seq_len for x in result["input_ids"]])
        assert all([len(x) == self.seq_len for x in result["attention_mask"]])
        assert all([len(x) == self.seq_len for x in result["labels"]])

        return result

    # Only Stride function
    # Takes batches at length seq_len, moving every stride
    # Masks out tokens that are reused the next batch
    def group_stride(self, examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if self.stride < self.seq_len:
            total_length_use = ((total_length - self.seq_len + self.stride) // self.stride) * self.stride
            result = {
                k: [t[i: i + self.seq_len] for i in range(0, total_length_use, self.stride)]
                for k, t in concatenated_examples.items()}
        elif self.stride > self.seq_len:
            count_indice = 1
            count_length = total_length - self.seq_len
            while count_length >= self.stride + self.seq_len:
                count_indice += 1
                count_length = count_length - self.stride - self.seq_len
            total_length_use = count_indice * self.stride
            result = {
                k: [t[0:self.seq_len]] for k, t in concatenated_examples.items()}
            result_add = {
                k: [t[i + self.stride + 1: i + self.seq_len + self.stride + 1] for i in range(self.stride, total_length_use, self.stride)]
                for k, t in concatenated_examples.items()}
            for key in result.keys():
                t = result[key]
                t1 = [item for sublist in [t, result_add[key]] for item in sublist]
                result[key] = t1
        # Copies over input ids to new column called labels
        result["labels"] = deepcopy(result["input_ids"])

        # Mask out losses in overlapping regions
        # Changes masked labels to -100 and attention mask to 0
        for i, labels in enumerate(result["labels"]):
            # Skip the first index since the first batch will not have any masking
            if i == 0:
                continue
            # For every j in range from 0 to length-stride, label to -100 to mask them
            for j in range(self.seq_len - self.stride):
                labels[j] = -100
            # Set the newly masked list of labels to result Dict object
            result["labels"][i] = labels

        for i, attention in enumerate(result["attention_mask"]):
            # Skip the first index since the first batch will not have any masking
            if i == 0:
                continue
            # For every j in range from 0 to length-stride, label to -100 to mask them
            for j in range(self.seq_len - self.stride):
                attention[j] = 0
            # Set the newly masked list of labels to result Dict object
            result["attention_mask"][i] = attention

        # Some checks

        #assert all([len(x) == self.seq_len for x in result["input_ids"]])
        #assert all([len(x) == self.seq_len for x in result["attention_mask"]])
        #assert all([len(x) == self.seq_len for x in result["labels"]])

        return result

    # Padding and stride function
    def group_padding_stride(self, examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # Finds just the quotient of total_length - seq_len by stride
        total_length_stride = ((total_length - self.seq_len + self.stride) // self.stride) * self.stride
        # Get the remainder and subtract to get the length of padding to add to fit the last stride
        # Different padding for stride > or < than seq_len
        if self.stride < self.seq_len:
            remainder = (total_length - self.seq_len) % self.stride
            to_add = self.seq_len - remainder
            to_add_input_id = [self.padding_tok] * to_add
            to_add_atten_mask = [0] * to_add
            pad_dict = dict(input_ids=to_add_input_id, attention_mask=to_add_atten_mask)
            for key in concatenated_examples.keys():
                t = concatenated_examples[key]
                t1 = [item for sublist in [t, pad_dict[key]] for item in sublist]
                # assert not len(t1) % self.stride
                concatenated_examples[key] = t1
            total_length_use = total_length_stride + 1
            # New Dict object based that samples at length seq_len with stride
            result = {k: [t[i: i + self.seq_len] for i in range(0, total_length_use, self.stride)] for k, t in
                      concatenated_examples.items()}
        elif self.stride > self.seq_len:
            count_index = 1
            count_length = total_length - self.seq_len
            while count_length >= self.stride + self.seq_len:
                count_index += 1
                count_length = count_length - self.stride - self.seq_len
            to_add = self.seq_len
            to_add_input_id = [self.padding_tok] * to_add
            to_add_atten_mask = [0] * to_add
            total_length_use = count_index * self.stride
            pad_dict = dict(input_ids=to_add_input_id, attention_mask=to_add_atten_mask)
            for key in concatenated_examples.keys():
                t = concatenated_examples[key]
                t1 = [item for sublist in [t, pad_dict[key]] for item in sublist]
                # assert not len(t1) % self.stride
                concatenated_examples[key] = t1

            # New Dict object based that samples at length seq_len with stride
            result = {k: [t[0:self.seq_len]] for k, t in concatenated_examples.items()}
            result_add = {
                k: [t[i + self.stride - 1: i + self.seq_len + self.stride - 1] for i in
                    range(self.stride, total_length_use, self.stride)]
                for k, t in concatenated_examples.items()}
            for key in result.keys():
                t = result[key]
                t1 = [item for sublist in [t, result_add[key]] for item in sublist]
                result[key] = t1

        # Copies over input ids to new column called labels
        result["labels"] = deepcopy(result["input_ids"])

        # Label is -100 if attention mask is 0, otherwise same as input ids
        # Just for padding at the end
        result["labels"] = [
            [-100 if mask == 0 else token for mask, token in mask_and_tokens] for mask_and_tokens in
            [zip(masks, labels) for masks, labels in zip(result["attention_mask"], result["labels"])]
        ]

        # Mask out losses in overlapping regions. If training data, string will be equal to seq_len
        for i, labels in enumerate(result["labels"]):
            # Skip the first index since the first batch will not have any masking
            if i == 0:
                continue
            # For every j in range from 0 to length-stride, change label to -100 to mask them
            for j in range(self.seq_len - self.stride):
                labels[j] = -100
            # Set the newly masked list of labels to result Dict object
            result["labels"][i] = labels

        for i, attention in enumerate(result["attention_mask"]):
            # Skip the first index since the first batch will not have any masking
            if i == 0:
                continue
            # For every j in range from 0 to length-stride, change attention mask to 0 to mask them
            for j in range(self.seq_len - self.stride):
                attention[j] = 0
            # Set the newly masked list of labels to result Dict object
            result["attention_mask"][i] = attention

        # Some checks
        # assert all([len(x) == self.seq_len for x in result["input_ids"]])
        # assert all([len(x) == self.seq_len for x in result["attention_mask"]])
        # assert all([len(x) == self.seq_len for x in result["labels"]])

        return result

    # If-else function calls based on padding and stride values of self
    def get_grouping(self, examples):
        # Split function calls by the inputs
        if self.padding is False and self.stride is self.seq_len:
            return self.group_default(examples)
        elif self.padding is True and self.stride is self.seq_len:
            return self.group_padding(examples)
        elif self.padding is False and self.stride is not self.seq_len:
            return self.group_stride(examples)
        elif self.padding is True and self.stride is not self.seq_len:
            return self.group_padding_stride(examples)


def collate_fn(all_data):
    keys = list(all_data[0].keys())
    ret_dict = {}
    for other_key in keys:
        all_other_value = [torch.LongTensor(_data[other_key]) for _data in all_data]
        all_other_value = torch.stack(all_other_value, 0)
        ret_dict[other_key] = all_other_value
    return ret_dict
