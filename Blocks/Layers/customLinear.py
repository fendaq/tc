import torch as t


class CustomLinear(t.nn.Module):

    def __init__(self, in_feature, out_feature, bias=True, bn=True, act=t.nn.ReLU()):
        super(CustomLinear, self).__init__()
        self.bn = bn
        self.linear = t.nn.Linear(in_feature, out_feature, bias)
        t.nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            t.nn.init.constant_(self.linear.bias, 0)
        if bn:
            self.batch_norm = t.nn.BatchNorm1d(out_feature)
        if act is not None:
            self.act = act

    def forward(self, inputs):
        net = self.linear(inputs)
        if self.bn:
            net = self.batch_norm(net.transpose(-2, -1)).transpose(-2, -1)
        if self.act is not None:
            net = self.act(net)
        return net

# a = t.randn((64,100,300))
# cl = CustomLinear(300,100)
# b = cl(a)