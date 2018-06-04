import torch as t
from torchnet import meter
import fire
from Models import TextCnn
from Configs import DefaultConfig
import pickle as pk
from DataSets import DataSet
from torch.utils.data import DataLoader
import ipdb


def logits_pre(acc):
    pass

def train(**kwargs):

    # init
    args = DefaultConfig()
    args.parse(kwargs)
    processor = pk.load(open('pr.pkl', 'rb'))
    seg_vocab_size = len(processor.token2id['seg'])
    char_vocab_size = len(processor.token2id['char'])
    model = TextCnn(args, seg_vocab_size, char_vocab_size)

    # dataset
    train_set = DataSet(args.char_max_lenth, args.word_max_lenth,'processed/data_train/',processor)
    valid_set = DataSet(args.char_max_lenth, args.word_max_lenth,'processed/data_valid/',processor)
    test_set = DataSet(args.char_max_lenth, args.word_max_lenth,'processed/data_test/',processor)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_set, args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, args.batch_size, shuffle=True, drop_last=True)

    # loss meter
    bceloss = t.nn.BCEWithLogitsLoss()
    mseloss = t.nn.MSELoss()
    celoss = t.nn.CrossEntropyLoss()

    acc_loss_meter = meter.AverageValueMeter()
    law_loss_meter = meter.AverageValueMeter()
    imprison_loss_meter = meter.MSEMeter()
    lr = args.lr
    optimizer = t.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(args.epochs):
        acc_loss_meter.reset()
        law_loss_meter.reset()
        imprison_loss_meter.reset()
        for step, datas in enumerate(train_loader):
            accusation_names, seg, pos, char, term_of_imprisonment, accusation, law = [i for i in datas]
            ipdb.set_trace()
            optimizer.zero_grad()
            accusation_logits, law_logits, imprison_logits = model(seg, accusation_names)
            acc_loss = bceloss(accusation_logits, accusation.float())
            law_loss = bceloss(law_logits, law.float())
            imprison_loss = mseloss(imprison_logits, term_of_imprisonment.unsqueeze(-1).float())
            loss = 0.4 * acc_loss + 0.4 * law_loss + 0.2 * imprison_loss
            loss.backward()
            optimizer.step()

train()
# #
# import torch as t
# from torchnet import meter
# import fire
# from Models import TextCnn
# from Configs import DefaultConfig
# import pickle as pk
# from DataSets import DataSet
# from torch.utils.data import DataLoader
#
# args = DefaultConfig()
# processor = pk.load(open('pr.pkl', 'rb'))
# seg_vocab_size = len(processor.token2id['seg'])
# char_vocab_size = len(processor.token2id['char'])
# model = TextCnn(args, seg_vocab_size, char_vocab_size)
# train_set = DataSet(args.char_max_lenth, args.word_max_lenth, 'processed/data_train/', processor)
# train_loader = DataLoader(train_set, args.batch_size, shuffle=True, drop_last=True)
#
# for i in train_loader:
#     cc = i
#     break
#
# dd = model(cc[1], cc[0],)
# #
# #
# #
