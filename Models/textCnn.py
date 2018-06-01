import torch as t
from Blocks.Layers import Highway, SelfAttention
from Blocks import EncoderLaw, EncoderAccusation, EncoderFact, Fusion
from Models import BasicModel
import ipdb
t.nn.Embedding()

class TextCnn(BasicModel):
    def __init__(self, args, seg_vocab_size, char_vocab_size):
        super(TextCnn, self).__init__()
        self.char_embedding_dim = args.char_embedding_dim
        self.seg_embedding_dim = args.seg_embedding_dim
        self.seg_vocab_size = seg_vocab_size
        self.char_vocab_size = char_vocab_size
        self.hidden_size = args.hidden_size
        self.num_head = args.num_head
        self.seg_embedding = t.nn.Sequential(t.nn.Embedding(self.seg_vocab_size,
                                                            self.seg_embedding_dim),
                                             Highway(input_dim=self.seg_embedding_dim,
                                                     num_layers=2))
        self.char_embedding = t.nn.Sequential(t.nn.Embedding(self.char_vocab_size,
                                                             self.char_embedding_dim),
                                              Highway(input_dim=self.char_embedding_dim,
                                                      num_layers=2))
        self.EncoderLaw = EncoderLaw(self.seg_embedding_dim, self.hidden_size, self.hidden_size)
        self.EncoderFact = EncoderFact(self.seg_embedding_dim, self.hidden_size, self.num_head, self.hidden_size)
        self.EncoderAccusation = EncoderLaw(self.seg_embedding_dim, self.hidden_size, self.hidden_size)

        self.Fusion = Fusion(self.seg_embedding_dim, self.seg_embedding_dim, self.seg_embedding_dim, self.hidden_size)
        self.SelfAttention = SelfAttention(self.seg_embedding_dim, self.hidden_size)

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
        law = self.char_embedding(law_char)
        accusation = self.char_embedding(accusation_char)
        # encodes
        encoded_fact = self.EncoderFact(fact, fact_mask)
        encoded_law = self.EncoderLaw(law, law_mask)
        encoded_accusation = self.EncoderAccustaion(accusation, accusation_mask)
        # fusions
        net = self.Fusion(encoded_fact, encoded_law, encoded_accusation)
        # self attention
        net = self.SelfAttention(net, fact_mask)
        return net
        # # decode
        # accusation_logits, law_logits, imprison_logits = Decoder(net)
        # return accusation_logits, law_logits, imprison_logits
        #

import pickle as pk
pr = pk.load(open('pr.pkl', 'rb'))
from Configs import DefaultConfig
args = DefaultConfig

fact_seg = t.ones((64,100)).long()
law_char = t.ones((64,202,10)).long()
accusation = t.ones((64,189,15)).long()
model = TextCnn(args,len(pr.token2id['seg']),len(pr.token2id['char']))
net = model(fact_seg,law_char,accusation)