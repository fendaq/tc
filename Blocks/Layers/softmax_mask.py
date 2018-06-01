from torch.nn.functional import softmax


def softmax_mask(vector, mask, dim):
    """

    :param vector: float [[1.3,2,-1.9],[2.3,-1.2,-3.1]]
    :param mask: float [[1,1,0],[1,1,1]
    :param dim: softmax dim
    :return: 
    """
    mask = (1 - mask) * -1e20
    masked_vector = vector + mask
    softmaxed_masked_vector = softmax(masked_vector, dim)
    return softmaxed_masked_vector
