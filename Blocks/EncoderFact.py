import torch as t
from Blocks.Layers import MultiHeadSelfAttention, LayerNorm, CustomLinear


class EncoderFact(t.nn.Module):
    def __init__(self, embedding_dim, hidden_size, num_head, output_dim):
        super(EncoderFact, self).__init__()
        self.multiheadselfattention = MultiHeadSelfAttention(input_dim=embedding_dim,
                                                             hidden_size=hidden_size,
                                                             num_head=num_head,
                                                             output_dim=output_dim)
        self.layernorm = LayerNorm(output_dim)
        self.feedforward1 = CustomLinear(output_dim, output_dim)
        self.feedforward2 = CustomLinear(output_dim, output_dim)

    def forward(self, inputs, mask):
        """

        :param inputs: [batch, seq_len, law_len, emb]
        :param mask: [batch, seq_len, law_len]
        :return:
        """
        net = self.multiheadselfattention(inputs, mask)
        net = self.layernorm(net)
        net = self.feedforward1(net)
        net = self.feedforward2(net)
        return net


# fact = t.randn((64,100,300))
# mask = t.ones((64,100))
# fe = EncoderFact(300,100,3,150)
# fe(fact,mask).shape



# law = t.randn((64,202,20,300))
# law_mask = t.randn((64,202,20))
