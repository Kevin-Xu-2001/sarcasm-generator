import torch
from torch.utils.data import dataset
import os
import json
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils import get_tokenizer
from tqdm import tqdm


class GPT2Dataset(Dataset):
    """Dataset class to feed the data from given directory to GPT2 model."""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.tokenizer = get_tokenizer()
        self.dataset = []

        df = pd.read_csv(data_dir)
        df = df[df['label'] == 1]
        df = df[df['score'] > 30]
        df.reset_index(drop=True, inplace=True)
        df.drop(
            labels=[
                'label',
                'author',
                'subreddit',
                'ups',
                'downs',
                'date',
                'created_utc'],
            axis=1,
            inplace=True)
        df['len_comment'] = [len(str(x)) for x in df['comment']]
        df['len_parent'] = [len(str(x)) for x in df['parent_comment']]
        df = df[df['len_parent'] < 1000]
        df = df[df['len_comment'] < 1000]

        for _, row in tqdm(df.iterrows(), total=df.shape[0]):
            try:
                reply_raw = row['comment']
                parent_raw = row['parent_comment']

                # tokenize and check if the (len(parent_tokenized) + len(reply_tokenized)) <= 1022
                # 1022 because 2 tokens have to be reserved for <|sep|> and
                # <|eos|>

                parent, reply = self.tokenizer.encode(
                    parent_raw), self.tokenizer.encode(reply_raw)

                if len(parent) > 0 and len(reply) > 0 and (
                        len(parent) + len(reply)) <= 1022:
                    self.dataset.append({
                        'parent': parent,
                        'reply': reply
                    })
            except BaseException:
                continue

        # Calculate length while instantiating becasue it might not be
        # efficient to do so multiple times from scratch
        self.length = len(self.dataset)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # load the data at given index from dataset
        data = self.dataset[idx]

        # initialize context with 1024 <|pad|> tokens since the input size of
        # GPT2 is 1024
        context = self.tokenizer.encode(self.tokenizer.pad_token) * 1024

        # make the required context by concatenating parent_comment + <|sep|> + reply_comment + <|eos|>
        # update that in the context
        text = data['parent'] + self.tokenizer.encode(
            self.tokenizer.sep_token) + data['reply'] + self.tokenizer.encode(self.tokenizer.eos_token)
        # this replaces the first len(text) tokens of the int list with the
        # appropriate text tokens, and remaining ones are still <|pad|> as
        # required
        context[:len(text)] = text

        # convert the context into a pyTorch tensor
        context = torch.tensor(context)

        # return the context along with the location of <|sep|> token so that the loss can be calculated only over the
        # reply part of the context i.e., after <|sep|> token
        return {'context': context, 'loc_sep': len(data['parent'])}
