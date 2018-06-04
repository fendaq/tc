import torch as t
from torchnet import meter
import fire
from Models import TextCnn
from Configs import DefaultConfig
import pickle as pk
from DataSets import DataSet
from torch.utils.data import DataLoader


def train_step(model,):


def train(**kwargs):

    # init
    args = DefaultConfig()
    args.parse(kwargs)
    processor = pk.load(open('pk.pkl', 'rb'))
    seg_vocab_size = len(processor.token2id['seg'])
    char_vocab_size = len(processor.token2id['char'])
    model = TextCnn(args, seg_vocab_size, char_vocab_size)

    # dataset
    train_set = DataSet(args.char_max_lenth, args.word_max_lenth)
    valid_set = DataSet(args.char_max_lenth, args.word_max_lenth)
    test_set = DataSet(args.char_max_lenth, args.word_max_lenth)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_set, args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, args.batch_size, shuffle=True, drop_last=True)

    # loss meter
    bceloss = t.nn.BCEWithLogitsLoss()
    mseloss = t.nn.MSELoss()
    celoss = t.nn.CrossEntropyLoss()

