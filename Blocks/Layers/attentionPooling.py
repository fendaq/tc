import torch as t
from Blocks.Layers import CustomLinear, softmax_mask
import ipdb


class AttentionPooling(t.nn.Module):
    def __init__(self, input_dim, hidden_size):
        super(AttentionPooling, self).__init__()
        self.linear1 = CustomLinear(input_dim, hidden_size, act=t.nn.Tanh())
        self.linear2 = CustomLinear(hidden_size, 1, bias=False,bn=False,act=None)

    def forward(self, inputs, mask):
        net = self.linear1(inputs)
        net = self.linear2(net)
        alpha = softmax_mask(net.squeeze(), mask, -1)
        alpha = alpha.unsqueeze(-1)
        res = alpha * inputs
        res = res.sum(1)
        return res

# inputs = t.randn((64,100,300))
# mask = t.ones(64,100)
# ap = AttentionPooling(300,100)
# cc = ap(inputs,mask)