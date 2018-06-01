#import thulac
from jieba.posseg import POSTokenizer
import json
import ipdb
import os
from tqdm import tqdm
import shutil
from collections import Counter
import pickle as pk
import fire


class Preprocessor(object):
    def __init__(self, root_raw='raw/', root_split='processed/', model=POSTokenizer()):
        self.model = model
        self.root_raw = root_raw
        self.root_split = root_split
        self.accus = {str.rstrip(accu): index for index, accu in enumerate(open('raw/accu.txt', 'r').readlines())}
        self.laws = {str.rstrip(law): index for index, law in enumerate(open('raw/law.txt', 'r').readlines())}
        self.counter = {'seg': Counter(), 'pos': Counter(), 'char': Counter()}
        self.file_dict = {'train': root_raw+'data_train.json', 'dev': root_raw+'data_valid.json', 'test': root_raw+'data_test.json'}
        self.label_dict = {'accusation': 'raw/accu.txt', 'law': 'raw/law.txt'}
        self.token2id = {'seg': {'<PAD>': 0, '<UNK>': 1}, 'pos': {'<PAD>': 0, '<UNK>': 1}, 'char': {'<PAD>': 0, '<UNK>': 1}}

    def process_content(self, contents):
        root = 'contents/'
        if os.path.exists(root):
            shutil.rmtree(root)
        os.mkdir(root)

        for index, content in enumerate(contents):
            result = self.process_one(index, content, meta=None, convert_to_id=True, build_counter=False)
            file_name = root+str(index)+'.json'
            json.dump(result, open(file_name, 'w'))

    def process_file_pipline(self, wv_seg=None, wv_char=None):
        self.process_file(task='build_vocab')
        self.build_vocab(wv_seg, wv_char)
        self.process_file(task='pure_process')

    def process_file(self, task='build_vocab'):
        """
        :param task: str [build_vocab, pure_process]
        :return: 
        """
        if os.path.exists(self.root_split):
            shutil.rmtree(self.root_split)
        os.mkdir(self.root_split)
        for file in list(self.file_dict.values()):

            folder_name = self.root_split+str(file.split('/')[-1].split('.')[0])+'/'
            os.mkdir(folder_name)

            for index, str_line in tqdm(enumerate(open(file, 'r')), desc=folder_name):
                json_line = json.loads(str_line)
                if task == 'build_vocab':
                    self.process_one(index, json_line['fact'], json_line['meta'], convert_to_id=False, build_counter=True)

                elif task == 'pure_process':
                    result = self.process_one(index, json_line['fact'], json_line['meta'], convert_to_id=True, build_counter=False)
                    file_name = folder_name + str(index) + '.json'
                    json.dump(result, open(file_name, 'w'))

    def process_one(self, index, fact, meta=None, convert_to_id=False, build_counter=False):
        # fact: str '我爱北京天安门'
        result = {'feature':{},'label':{}}
        result['index'] = index
        token_list = [[word, seg] for word, seg in self.model.cut(fact)]
        # token_list : [['seg','pos'],]
        if not convert_to_id:
            result['feature']['seg'] = [i[0] for i in token_list]
            result['feature']['pos'] = [i[1] for i in token_list]
            result['feature']['char'] = [i for i in fact]
        else:
            result['feature']['seg'] = [self.token2id['seg'][i[0]] if i[0] in self.token2id['seg'] else 1 for i in token_list]
            result['feature']['pos'] = [self.token2id['pos'][i[1]] if i[1] in self.token2id['pos'] else 1 for i in token_list]
            result['feature']['char'] = [self.token2id['char'][i] if i in self.token2id['char'] else 1 for i in fact]
        if meta is not None:
            result['label']['term_of_imprisonment'] = -2 if meta['term_of_imprisonment']['death_penalty'] \
                else -1 if meta['term_of_imprisonment']['life_imprisonment'] \
                else meta['term_of_imprisonment']['imprisonment']
            accusation_id = [self.accus[i] for i in meta['accusation']]
            law_id = [self.laws[str(i)] for i in meta['relevant_articles']]
            result['label']['accusation'] = [1 if i in accusation_id else 0 for i in range(len(self.accus))]
            result['label']['law'] = [1 if i in law_id else 0 for i in range(len(self.laws))]

        if build_counter:
            self.counter['seg'].update(Counter(result['feature']['seg']))
            self.counter['pos'].update(Counter(result['feature']['pos']))
            self.counter['char'].update(Counter(result['feature']['char']))
        return result

    def build_vocab(self, wv_seg=None, wv_char=None):
        for key in list(self.counter.keys()):
            idx = 2
            for item in tqdm(self.counter[key].items(), desc='building '+key):
                if item[1] > 1:
                    self.token2id[key][item[0]] = idx
                    idx += 1
        # todo use pretrained

    def save(self):
        pk.dump(self, open('pr.pkl', 'wb'))
        print('saved')
