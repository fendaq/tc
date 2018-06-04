import math
import torch as t
from Blocks.Layers import CustomLinear
from Blocks.Layers.softmax_mask import softmax_mask
import ipdb


class DotAttention(t.nn.Module):
    def __init__(self, query_dim, passage_dim, hidden_size):
        super(DotAttention, self).__init__()
        self.query_dim = query_dim
        self.passage_dim = passage_dim
        self.hidden_size = hidden_size
        self.linear_query = CustomLinear(query_dim, hidden_size, True, True, t.nn.ReLU())
        self.linear_passage = CustomLinear(passage_dim, hidden_size, True, True, t.nn.ReLU())

    def forward(self, query, passage, query_mask, passage_mask):
        """
        
        :param query: [b,lq,h]
        :param passage: [b,lp,h]
        :param query_mask: [b,lq]
        :param passage_mask: [b,lp]
        :return: [b,lp,h]
        """
        query_mask = query_mask.unsqueeze(-2)
        passage_mask = passage_mask.unsqueeze(-1)
        mask = passage_mask.bmm(query_mask)
        query = self.linear_query(query).transpose(-1, -2)
        passage = self.linear_passage(passage)
        dot = passage.bmm(query)/math.sqrt(self.hidden_size)
        masked_dot = softmax_mask(dot, mask, -1)
        outputs = masked_dot.bmm(query.transpose(-2, -1))
        outputs = outputs + passage
        return outputs


class DotAttentionGated(t.nn.Module):
    def __init__(self, query_dim, passage_dim, hidden_size):
        super(DotAttentionGated, self).__init__()
        self.query_dim = query_dim
        self.passage_dim = passage_dim
        self.hidden_size = hidden_size
        self.linear_query = CustomLinear(query_dim, hidden_size, True, True, t.nn.ReLU())

        self.linear_passage = CustomLinear(passage_dim, hidden_size, True, True, t.nn.ReLU())
        self.gate = CustomLinear(hidden_size, hidden_size, bias=False, bn=True, act=t.nn.Sigmoid())

    def forward(self, query, passage, query_mask, passage_mask):
        """

        :param query: [b,lq,h]
        :param passage: [b,lp,h]
        :param query_mask: [b,lq]
        :param passage_mask: [b,lp]
        :return: [b,lp,h]
        """
        # mask
        query_mask = query_mask.unsqueeze(-2)
        passage_mask = passage_mask.unsqueeze(-1)
        mask = passage_mask.bmm(query_mask)
        # dot product
        query = self.linear_query(query).transpose(-1, -2)
        passage = self.linear_passage(passage)
        dot = passage.bmm(query) / math.sqrt(self.hidden_size)
        masked_dot = softmax_mask(dot, mask, -1)
        # attention dot query
        outputs = masked_dot.bmm(query.transpose(-2, -1))
        # gate
        gate = self.gate(outputs)
        outputs = passage * gate + (1-gate) * outputs
        return outputs


# query = t.randn((64, 100, 300))
# passage = t.randn((64, 120, 300))
# query_mask = t.ones((64, 100))
# passage_mask = t.ones((64, 120))
# dt = DotAttention(300, 300, 300)
# dt(query, passage, query_mask, passage_mask)
# #
# query = t.randn((64, 100, 300))
# passage = t.randn((64, 120, 300))
# query_mask = t.ones((64, 100))
# passage_mask = t.ones((64, 120))
# dt = DotAttentionGated(300, 300, 300)
# dt(query, passage, query_mask, passage_mask)