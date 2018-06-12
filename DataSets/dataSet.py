from torch.utils.data import Dataset
import os
import ipdb
import json
import numpy as np
import jieba


class DataSet(Dataset):
    def __init__(self, char_max_lenth, word_max_lenth, root, processor, word_features=['seg', 'pos'], char_features=['char'], labels=['term_of_imprisonment','accusation','law'],):
        """
        :param args: obj args
        :param root: str data root
        :param word_features: list ['seg','pos']
        :param char_features: list ['char']
        :param labels: list ['term_of_imprisonment','accusation','law']
        """
        super(DataSet, self).__init__()
        self.accus_char_maxlen = 32
        self.char_max_lenth = char_max_lenth
        self.word_max_lenth = word_max_lenth
        self.root = root
        self.file_list = [root+i for i in os.listdir(self.root)]
        self.word_features = word_features
        self.char_features = char_features
        self.labels = labels
        #self.law_feature = self.get_law_feature()
        self.accusation_feature = self.get_accusation_feature(processor)

    def get_law_feature(self):
        laws = [str.strip(i) for i in open('raw/law.txt').readlines()]
        seg_laws = [list(jieba.cut(i)) for i in laws]

    def get_accusation_feature(self, processor):
        accus = [str.strip(i) for i in open('raw/accu.txt').readlines()]
        char_accus = [[j for j in i] for i in accus]
        char_accus = [[processor.token2id['char'][i] if i in processor.token2id['char'] else 1 for i in accu] for accu in char_accus]
        char_accus = [self.pre_pad(i, self.accus_char_maxlen) for i in char_accus]
        return char_accus

    def pre_pad(self, feature, pad_to):
        ori_len = len(feature)
        if ori_len >= pad_to:
            padded = feature[-pad_to:]
        else:
            padded = feature+[0]*(pad_to-ori_len)
        return padded

    def get_return_list(self):
        sample = {}
        with open(self.file_list[0], 'r') as file:
            line = json.loads(file.readline())
        for feature in self.word_features:
            sample[feature] = self.pre_pad(line['feature'][feature], self.word_max_lenth)
        for feature in self.char_features:
            sample[feature] = self.pre_pad(line['feature'][feature], self.char_max_lenth)
        for label in self.labels:
            sample[label] = line['label'][label]
        return ['accusation']+[key for key in sample.keys()]

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
        return tuple([np.array(self.accusation_feature)] + [np.array(sample[key]) for key in sample.keys()])

    def __len__(self):
        return len(self.file_list)

#
# import pickle as pk
#
# processor = pk.load(open('pr.pkl', 'rb'))
#
# ds = DataSet(100,500,'processed/data_train/', processor)
# from torch.utils.data import DataLoader
#
# dl = DataLoader(ds,2)
# for i in dl:
#     cc = i
#     break
