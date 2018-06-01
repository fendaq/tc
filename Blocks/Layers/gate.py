import torch as t
from Blocks.Layers import CustomLinear


class Gate(t.nn.Module):
    def __init__(self, input_dim):
        super(Gate, self).__init__()
        self.gate = CustomLinear(input_dim, input_dim, bias=False, bn=True, act=t.nn.Sigmoid())
        self.nonlinear_input = CustomLinear(input_dim, input_dim, bias=True, bn=True, act=t.nn.ReLU())

    def forward(self, inputs):

        nonlinear = self.nonlinear_input(inputs)
        gate = self.gate(nonlinear)
        outputs = inputs * gate + (1-gate) * nonlinear
        return outputs
