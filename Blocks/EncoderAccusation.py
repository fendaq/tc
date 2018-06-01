import torch as t
from allennlp.modules.seq2vec_encoders import CnnEncoder


class EncoderAccusation(t.nn.Module):
    def __init__(self, embedding_dim, num_filter, output_dim):
        super(EncoderAccusation, self).__init__()
        self.output_dim = output_dim
        self.ce = CnnEncoder(embedding_dim, num_filter, output_dim=output_dim)

    def forward(self, inputs, mask):
        """

        :param inputs: [batch, seq_len, law_len, emb]
        :param mask: [batch, seq_len, law_len]
        :return: 
        """
        batch_size, seq_len, law_len, emb = inputs.size()
        inputs = inputs.view((-1, law_len, emb))
        mask = mask.view((-1, law_len))
        net = self.ce(inputs, mask)
        net = net.view((batch_size, seq_len, self.output_dim))
        return net


#
#
# ce = CnnEncoder(300,100)
# a = t.randn((64,30,100,300))


#
#
#
# law = t.randn((64,202,20,300))
# law_mask = t.randn((64,202,20))
