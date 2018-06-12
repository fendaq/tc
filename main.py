import torch as t
from torchnet import meter
import fire
from Models import TextCnn
from Configs import DefaultConfig
import pickle as pk
from DataSets import DataSet
from tqdm import tqdm
from torch.utils.data import DataLoader
import ipdb
from sklearn.metrics import confusion_matrix
import time

def score_sigmoid(pre_logit, tru):
    pre = t.nn.functional.sigmoid(pre_logit) > 0.5
    fns, fps, tns, tps, f1s = 0, 0, 0, 0, 0
    for i in zip(tru, pre):
        tn, fp, fn, tp = confusion_matrix(i[0], i[1]).ravel()
        tps += tp
        tns += tn
        fps += fp
        fns += fn
        precision = tp/(tp+fp+1e-8)
        recall = tp/(tp+fn+1e-8)
        f1 = 2*precision*recall/(precision+recall+1e-8)
        f1s += f1
    micro_precision = tps/(tps+fps+1e-8)
    micro_recall = tps/(tps+fns+1e-8)
    micro_f1 = 2*micro_precision*micro_recall/(micro_precision+micro_recall+1e-8)
    macro_f1 = f1s/len(tru)
    score = (micro_f1+macro_f1)/2
    return score


# def score_topk(pre_logit, tru, k):
#     pre = t.topk(pre_logit, k)[1]

def val(model, dev_loader):
    model.eval()
    bceloss = t.nn.BCEWithLogitsLoss()
    acc_loss_meter = meter.AverageValueMeter()
    law_loss_meter = meter.AverageValueMeter()
    acc_score_meter = meter.AverageValueMeter()
    law_score_meter = meter.AverageValueMeter()

    for step, datas in tqdm(enumerate(dev_loader)):
        accusation_names, seg, pos, char, term_of_imprisonment, accusation, law = [i.cuda() for i in datas]
        accusation_logits, law_logits, imprison_logits = model(seg, accusation_names)
        acc_loss = bceloss(accusation_logits, accusation.float())
        law_loss = bceloss(law_logits, law.float())
        acc_score = score_sigmoid(accusation_logits, accusation)
        law_score = score_sigmoid(law_logits, law)
        loss = 0.5 * acc_loss + 0.5 * law_loss  # + 0.2 * imprison_loss
        acc_loss_meter.add(acc_loss.item())
        law_loss_meter.add(law_loss.item())
        acc_score_meter.add(acc_score)
        law_score_meter.add(law_score)
    model.train()
    return acc_loss_meter.value()[0], law_loss_meter.value()[0], acc_score_meter.value()[0], law_score_meter.value()[0]

def train(**kwargs):

    # init
    print('init')
    args = DefaultConfig()
    args.parse(kwargs)
    processor = pk.load(open('pr.pkl', 'rb'))
    seg_vocab_size = len(processor.token2id['seg'])
    char_vocab_size = len(processor.token2id['char'])
    model = TextCnn(args, seg_vocab_size, char_vocab_size, processor.seg_matrix, processor.char_matrix).cuda()
    print('build dataset')
    # dataset
    train_set = DataSet(args.char_max_lenth, args.word_max_lenth, 'processed/data_train/', processor)
    valid_set = DataSet(args.char_max_lenth, args.word_max_lenth, 'processed/data_valid/', processor)
    test_set = DataSet(args.char_max_lenth, args.word_max_lenth, 'processed/data_test/', processor)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_set, args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, args.batch_size, shuffle=True, drop_last=True)

    # loss meter
    bceloss = t.nn.BCEWithLogitsLoss()
    mseloss = t.nn.MSELoss()
    celoss = t.nn.CrossEntropyLoss()

    acc_loss_meter = meter.AverageValueMeter()
    law_loss_meter = meter.AverageValueMeter()
    #imprison_loss_meter = meter.MSEMeter()
    lr = args.lr
    optimizer = t.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    print('start')
    best_score = -1.0
    for epoch in range(args.epochs):
        acc_loss_meter.reset()
        law_loss_meter.reset()
        #imprison_loss_meter.reset()
        for step, datas in tqdm(enumerate(train_loader), desc='step:'):
            accusation_names, seg, pos, char, term_of_imprisonment, accusation, law = [i.cuda() for i in datas]
            optimizer.zero_grad()
            accusation_logits, law_logits, imprison_logits = model(seg, accusation_names)
            acc_loss = bceloss(accusation_logits, accusation.float())
            law_loss = bceloss(law_logits, law.float())
            #imprison_loss = mseloss(imprison_logits, term_of_imprisonment.unsqueeze(-1).float())
            loss = 0.5 * acc_loss + 0.5 * law_loss# + 0.2 * imprison_loss
            loss.backward()
            optimizer.step()
            if (step % 500 == 0) & (step != 0):
                acc_score = score_sigmoid(accusation_logits, accusation)
                law_score = score_sigmoid(law_logits, law)
                val_acc_loss, val_law_loss, val_acc_score, val_law_score = val(model, valid_loader)
                print('epoch:%s,step:%s'%(epoch, step))
                print('    train:')
                print('        loss:%s,accsocre:%s,lawscore:%s'%(loss.item(), acc_score, law_score))
                print('    val:')
                print('        accloss:%s,law_loss:%s,accscore:%s,lawscore:%s' % (val_acc_loss, val_law_loss, val_acc_score, val_law_score))
                print(' ')
        val_acc_loss, val_law_loss, val_acc_score, val_law_score = val(model, valid_loader)
        print('epoch:', epoch)
        print('    train:')
        print('        accloss:%s,law_loss:%s' % (acc_loss_meter.value()[0], law_loss_meter.value()[0]))
        print('    val:')
        print('        accloss:%s,law_loss:%s,accscore:%s,lawscore:%s' % (val_acc_loss, val_law_loss, val_acc_score, val_law_score))
        score = val_law_score+val_acc_score
        if score > best_score:
            model.save()
#        if 0.5*val_acc_loss+0.5*val_law_loss >

# TODO: lr decay
# TODO: model save
# TODO: predictor


if __name__ == '__main__':
    fire.Fire()
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
import torchtext.data