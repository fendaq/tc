import torch as t
from Blocks.Layers import CustomLinear
from Blocks.Layers import softmax_mask
import math
import ipdb


class MultiHeadSelfAttention(t.nn.Module):
    def __init__(self, input_dim, hidden_size, num_head, output_dim):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_head = num_head
        self.hidden_size = hidden_size
        self.passage_linear = CustomLinear(input_dim, hidden_size)
        self.query_linear = CustomLinear(input_dim, hidden_size)
        self.output_linear = CustomLinear(num_head*hidden_size, output_dim)

    def forward(self, inputs, mask):
        """
        :param inputs: 
        :param mask: 
        :return: 
        """
        mask = mask.unsqueeze(-1).bmm(mask.unsqueeze(-2))
        batch_size, seq_len, input_dim = inputs.size()
        # [num_head,batch,seq_len, input_dim]
        multihead_inputs = inputs.repeat(self.num_head, 1, 1)
        # [num_head, batch*seq_len, input_dim]
        multihead_inputs = multihead_inputs.view(self.num_head, batch_size * seq_len, input_dim)
        #  -> [num_head, batch*seq_len, hidden_size]
        #  -> [num_head*batch, seq_len, hidden_size]
        passage = self.passage_linear(multihead_inputs).view(self.num_head * batch_size, seq_len, self.hidden_size)
        query = self.query_linear(multihead_inputs).view(self.num_head*batch_size, seq_len, self.hidden_size)
        # [num_head*batch, seq_len, seq_len]
        alpha = query.bmm(passage.transpose(-1, -2))/math.sqrt(self.hidden_size)
        # mask & normalize
        alpha = softmax_mask(alpha, mask.repeat(self.num_head, 1, 1), dim=-1)
        # outputs [num_head*batch, seq_len, hidden_size]
        outputs = alpha.bmm(passage)
        # split head  num_head * [batch, seq_len, hidden_size]
        outputs = t.split(outputs, batch_size, 0)
        # cat [batch, seq_len, hidden_size*num_head]
        outputs = t.cat(outputs, dim=-1)
        # [batch, seq_len, hidden_size]
        outputs = self.output_linear(outputs)
        return outputs

#
# inputs = t.randn((64, 400, 300))
# mask = t.ones((64, 400))
# ma = MultiHeadSelfAttention(300, 100, 3, 300)
