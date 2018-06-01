import torch as t
from collections import OrderedDict
import ipdb


class CnnEncoder(t.nn.Module):
    def __init__(self, input_dim, sub_out_dim, output_dim):
        super(CnnEncoder, self).__init__()
        self.conv1 = t.nn.Sequential(OrderedDict([
            ('conv', t.nn.Conv1d(input_dim, sub_out_dim, kernel_size=1)),
            ('batchnorm', t.nn.BatchNorm1d(sub_out_dim)),
            ('act', t.nn.ReLU())
        ]))
        self.conv2 = t.nn.Sequential(OrderedDict([
            ('conv', t.nn.Conv1d(input_dim, sub_out_dim, kernel_size=1)),
            ('batchnorm', t.nn.BatchNorm1d(sub_out_dim)),
            ('act', t.nn.ReLU()),
            ('conv2', t.nn.Conv1d(sub_out_dim, sub_out_dim, kernel_size=3)),
            ('batchnorm2', t.nn.BatchNorm1d(sub_out_dim)),
            ('act2', t.nn.ReLU()),
            ('conv3', t.nn.Conv1d(sub_out_dim, sub_out_dim, kernel_size=3)),
            ('batchnorm3', t.nn.BatchNorm1d(sub_out_dim)),
            ('act3', t.nn.ReLU())
        ]))
        self.conv3 = t.nn.Sequential(OrderedDict([
            ('conv', t.nn.Conv1d(input_dim, sub_out_dim, kernel_size=1)),
            ('batchnorm', t.nn.BatchNorm1d(sub_out_dim)),
            ('act', t.nn.ReLU()),
            ('conv2', t.nn.Conv1d(sub_out_dim, sub_out_dim, kernel_size=3)),
            ('batchnorm2', t.nn.BatchNorm1d(sub_out_dim)),
            ('act2', t.nn.ReLU())
        ]))
        self.conv4 = t.nn.Sequential(OrderedDict([
            ('max_pooling', t.nn.MaxPool1d(1, 1)),
            ('conv', t.nn.Conv1d(input_dim, sub_out_dim, 1))
        ]))
        self.linear = t.nn.Sequential(OrderedDict([
            ('linear', t.nn.Linear(sub_out_dim, output_dim)),
            ('batchnorm', t.nn.BatchNorm1d(output_dim)),
            ('act', t.nn.ReLU())
        ]))

    def forward(self, inputs, mask=None):
        # masking
        if mask is not None:
            inputs = inputs * mask.unsqueeze(-1).float()
        inputs = inputs.transpose(-1, -2)
        # convs
        convs1 = self.conv1(inputs)
        convs2 = self.conv2(inputs)
        convs3 = self.conv3(inputs)
        convs4 = self.conv4(inputs)
        net = t.cat([convs1, convs2, convs3, convs4], -1)
        # max pool
        net = net.max(-1)[0]
        # linear
        net = self.linear(net)
        return net
#
#
# a = t.randn((64,120,100))
# ce = CnnEncoder(100,100,300)
# ce(a)