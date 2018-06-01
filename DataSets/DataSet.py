import torch as t
from torch.utils.data import Dataset, DataLoader
import os
import json
import numpy as np


class DataSet(Dataset):
    def __init__(self, char_max_lenth, word_max_lenth, root, word_features=['seg', 'pos'], char_features=['char'], labels=['term_of_imprisonment','accusation','law']):
        """
        :param args: obj args
        :param root: str data root
        :param word_features: list ['seg','pos']
        :param char_features: list ['char']
        :param labels: list ['term_of_imprisonment','accusation','law']
        """
        super(DataSet, self).__init__()
        self.char_max_lenth = char_max_lenth
        self.word_max_lenth = word_max_lenth
        self.root = root
        self.file_list = [root+i for i in os.listdir(self.root)]
        self.word_features = word_features
        self.char_features = char_features
        self.labels = labels

    def pre_pad(self, feature, pad_to):
        ori_len = len(feature)
        if ori_len >= pad_to:
            padded = feature[::-1][:pad_to][::-1]
        else:
            padded = feature+[0]*(pad_to-ori_len)
        return padded

    def get_return_list(self):
        sample = {}
        with open(self.file_list[item], 'r') as file:
            line = json.loads(file.readline())
        for feature in self.word_features:
            sample[feature] = self.pre_pad(line['feature'][feature], self.word_max_lenth)
        for feature in self.char_features:
            sample[feature] = self.pre_pad(line['feature'][feature], self.char_max_lenth)
        for label in self.labels:
            sample[label] = line['label'][label]
        return [key for key in sample.keys()]

    def __getitem__(self, item):
        sample = {}
        with open(self.file_list[item], 'r') as file:
            line = json.loads(file.readline())
        for feature in self.word_features:
            sample[feature] = self.pre_pad(line['feature'][feature], self.word_max_lenth)
        for feature in self.char_features:
            sample[feature] = self.pre_pad(line['feature'][feature], self.char_max_lenth)
        for label in self.labels:
            sample[label] = line['label'][label]
        return tuple([np.array(sample[key]) for key in sample.keys()])

    def __len__(self):
        return len(self.file_list)
