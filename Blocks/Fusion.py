import torch as t
from Blocks.Layers import CustomLinear, DotAttention


class Fusion(t.nn.Module):
    def __init__(self, fact_dim, law_dim, accusation_dim, hidden_size):
        super(Fusion, self).__init__()
        self.fact_law_attention = DotAttention(fact_dim, law_dim, hidden_size)
        self.fact_accusation_attention = DotAttention(fact_dim, accusation_dim, hidden_size)
        self.fusion_nonlinear = CustomLinear(3 * fact_dim, fact_dim, bias=True, bn=True, act=t.nn.ReLU())
        self.gate = CustomLinear(3 * fact_dim, fact_dim, bias=False, bn=True, act=t.nn.Sigmoid())

    def forward(self, fact, law, accusation, fact_mask, law_mask, accusation_mask):

        self.law_info = self.fact_law_attention(fact, law, fact_mask, law_mask)
        self.accusation_info = self.fact_accusation_attention(fact, law, fact_mask, law_mask)
        fusion = t.cat((fact, self.law_info, self.accusation_info), -1)
        fusion_nonlinear = self.fusion_nonlinear(fusion)
        gate = self.gate(fusion)
        outputs = gate * fusion_nonlinear + (1 - gate) * fact
        return outputs


fact = t.randn((64, 100, 300))
law = t.randn((64, 100, 300))
accusation = t.randn((64, 100, 300))
fact_mask = t.ones(64,100)
law_mask = t.ones(64,100)
accusation_mask = t.ones(64,100)


fu = Fusion(300,300,300,300)
a = fu(fact,law,accusation,fact_mask,law_mask,accusation_mask)
# #
# a = t.randn((64,100,300))
# cl = CustomLinear(300,100)
# b = cl(a)