import torch as t
from Blocks.Layers import Highway, SelfAttention, Gate, CustomLinear
from Blocks import EncoderLaw, EncoderAccusation, EncoderFact, Fusion
from Models import BasicModel
from collections import OrderedDict
import ipdb


class TextCnn(BasicModel):
    def __init__(self, args, seg_vocab_size, char_vocab_size):
        super(TextCnn, self).__init__()
        self.accusation_num = 202
        self.law_num = 183
        self.char_embedding_dim = args.char_embedding_dim
        self.seg_embedding_dim = args.seg_embedding_dim
        self.seg_vocab_size = seg_vocab_size
        self.char_vocab_size = char_vocab_size
        self.hidden_size = args.hidden_size
        self.num_head = args.num_head
        self.seg_embedding = t.nn.Sequential(t.nn.Embedding(self.seg_vocab_size,
                                                            self.seg_embedding_dim),
                                             Gate(self.seg_embedding_dim))
        self.char_embedding = t.nn.Sequential(t.nn.Embedding(self.char_vocab_size,
                                                             self.char_embedding_dim),
                                              Gate(self.seg_embedding_dim))
        self.EncoderLaw = EncoderLaw(self.seg_embedding_dim, self.hidden_size, self.hidden_size)
        self.EncoderFact = EncoderFact(self.seg_embedding_dim, self.hidden_size, self.num_head, self.hidden_size)
        self.EncoderAccusation = EncoderAccusation(self.seg_embedding_dim, self.hidden_size, self.hidden_size)

        self.Fusion = Fusion(self.seg_embedding_dim, self.seg_embedding_dim, self.seg_embedding_dim, self.hidden_size)
        self.SelfAttention = SelfAttention(self.seg_embedding_dim, self.hidden_size)

        self.accusation_linear = t.nn.Sequential(OrderedDict([
            ('linear1', CustomLinear(self.seg_embedding_dim, self.hidden_size)),
            ('linear2', CustomLinear(self.hidden_size, self.accusation_num))
        ]))
        self.law_linear = t.nn.Sequential(OrderedDict([
            ('linear1', CustomLinear(self.seg_embedding_dim, self.hidden_size)),
            ('linear2', CustomLinear(self.hidden_size, self.law_num))
        ]))
        self.imprison_linear = t.nn.Sequential(OrderedDict([
            ('linear1', CustomLinear(self.seg_embedding_dim, self.hidden_size)),
            ('linear2', CustomLinear(self.hidden_size, 1))
        ]))

    def get_mask(self, seg):
        """
        get word level mask
        :param seg:  [batch, seq_len]/[batch, seg_lenth, char_lenth]
        :return: mask: [batch, seg_len]/[batch, seg_lenth, char_lenth]
        """
        mask = seg.ne(0).float().detach()
        return mask

    def forward(self, fact_seg, law_char, accusation_char):
        """
        
        :param fact_seg: [batch, seq_lenth]
        :param law_char: [batch, law_num, char_lenth]
        :param accusation_char: [batch, accusation_num, char_lenth]
        :return: 
        """
        ipdb.set_trace()
        # masks
        fact_mask = self.get_mask(fact_seg)
        law_mask = self.get_mask(law_char)
        accusation_mask = self.get_mask(accusation_char)
        # embeddings
        fact = self.seg_embedding(fact_seg)
        batch, law_num, char_len = law_char.size()
        batch, accusation_num, char_len = accusation_char.size()
        law = self.char_embedding(law_char.view(-1, char_len)).view(batch, law_num, char_len, self.char_embedding_dim)
        accusation = self.char_embedding(accusation_char.view(-1, char_len)).view(batch, accusation_num, char_len, self.char_embedding_dim)
        # encodes
        encoded_fact = self.EncoderFact(fact, fact_mask)
        encoded_law = self.EncoderLaw(law, law_mask)
        encoded_accusation = self.EncoderAccusation(accusation, accusation_mask)
        # fusions
        net = self.Fusion(encoded_fact, encoded_law, encoded_accusation)
        # self attention
        net = self.SelfAttention(net, fact_mask)
        accusation_logits = self.accusation_linear(net)
        law_logits = self.law_linear(net)
        imprison_logits = self.imprison_linear(net)
        return net


# import pickle as pk
# pr = pk.load(open('pr.pkl', 'rb'))
# from Configs import DefaultConfig
# args = DefaultConfig
#
# fact_seg = t.ones((64, 100)).long()
# law_char = t.ones((64, 202, 10)).long()
# accusation = t.ones((64, 189, 10)).long()
# model = TextCnn(args, len(pr.token2id['seg']), len(pr.token2id['char']))
# net = model(fact_seg, law_char, accusation)