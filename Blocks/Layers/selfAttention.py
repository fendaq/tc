import math
import torch as t
from Blocks.Layers import CustomLinear
from Blocks.Layers.softmax_mask import softmax_mask
import ipdb


class SelfAttention(t.nn.Module):
    def __init__(self, input_dim, hidden_size):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.linear_query = CustomLinear(input_dim, hidden_size, bias=True, bn=True, act=t.nn.ReLU())
        self.linear_passage = CustomLinear(input_dim, hidden_size, bias=True, bn=True, act=t.nn.ReLU())
        self.gate = CustomLinear(input_dim, input_dim, bias=False, bn=True, act=t.nn.Sigmoid())

    def forward(self, inputs, mask):
        # mask
        query_mask = mask.unsqueeze(-2)
        passage_mask = mask.unsqueeze(-1)
        mask = passage_mask.bmm(query_mask)
        # dot
        query = self.linear_query(inputs).transpose(-1, -2)
        passage = self.linear_passage(inputs)
        dot = passage.bmm(query)/math.sqrt(self.hidden_size)
        masked_dot = softmax_mask(dot, mask, -1)
        # attention dot query
        outputs = masked_dot.bmm(passage)
        gate = self.gate(outputs)
        outputs = gate * inputs + (1-gate) * outputs
        return outputs

