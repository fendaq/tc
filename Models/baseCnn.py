import torch as t
from Blocks.Layers import Highway, SelfAttention, Gate, CustomLinear, AttentionPooling
from Blocks import EncoderLaw, EncoderAccusation, EncoderFact, Fusion
from Models import BasicModel
from collections import OrderedDict
import ipdb


class BaseCnn(BasicModel):
    def __init__(self, args, seg_vocab_size, char_vocab_size, seg_matrix, char_matrix):
        super(BaseCnn, self).__init__()
        self.accusation_num = 202
        self.law_num = 183
        self.imprison_num = 53
        self.char_embedding_dim = args.char_embedding_dim
        self.seg_embedding_dim = args.seg_embedding_dim
        self.seg_vocab_size = seg_vocab_size
        self.char_vocab_size = char_vocab_size
        self.hidden_size = args.hidden_size
        self.num_head = args.num_head

        self.seg_embedding = t.nn.Embedding(self.seg_vocab_size, self.seg_embedding_dim)
        self.seg_embedding.weight.data.copy_(t.from_numpy(seg_matrix))
        self.seg_embedding.weight.requires_grad = False
        self.EncoderFact = EncoderFact(self.seg_embedding_dim, self.hidden_size, self.num_head, self.hidden_size)
        self.AttentionPooling = AttentionPooling(self.hidden_size, self.hidden_size)
        self.accusation_linear = t.nn.Sequential(OrderedDict([
            ('linear', t.nn.Linear(self.hidden_size, self.hidden_size)),
            ('bn', t.nn.BatchNorm1d(self.hidden_size)),
            ('act', t.nn.ReLU()),
            ('linear2', t.nn.Linear(self.hidden_size, self.accusation_num)),
        ]))
        self.law_linear = t.nn.Sequential(OrderedDict([
            ('linear', t.nn.Linear(self.hidden_size, self.hidden_size)),
            ('bn', t.nn.BatchNorm1d(self.hidden_size)),
            ('act', t.nn.ReLU()),
            ('linear2', t.nn.Linear(self.hidden_size, self.law_num)),
        ]))

    def get_mask(self, seg):
        """
        get word level mask
        :param seg:  [batch, seq_len]/[batch, seg_lenth, char_lenth]
        :return: mask: [batch, seg_len]/[batch, seg_lenth, char_lenth]
        """
        mask = seg.ne(0).float().detach()
        return mask

    def forward(self, fact_seg, accusation_char):
        """

        :param fact_seg: [batch, seq_lenth]
        :param law_char: [batch, law_num, char_lenth]
        :param accusation_char: [batch, accusation_num, char_lenth]
        :return: 
        """
        # masks
        fact_mask = self.get_mask(fact_seg)
        # embeddings
        fact = self.seg_embedding(fact_seg)
        # encodes
        encoded_fact = self.EncoderFact(fact, fact_mask)
        # fusions
        # self attention
        net = self.SelfAttention(encoded_fact, fact_mask)
        net = self.AttentionPooling(net, fact_mask)
        accusation_logits = self.accusation_linear(net)
        law_logits = self.law_linear(net)
        imprison_logits = self.imprison_linear(net)

        return accusation_logits, law_logits, imprison_logits