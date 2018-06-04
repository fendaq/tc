class DefaultConfig(object):
    batch_size = 64
    char_embedding_dim = 300
    seg_embedding_dim = 300
    hidden_size = 150
    num_head = 3
    char_max_lenth = 1000
    word_max_lenth = 500


    def parse(self, kwargs, print_=True):
        '''
        根据字典kwargs 更新 config参数
        '''
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise Exception("opt has not attribute <%s>" % k)
            setattr(self, k, v)

    def state_dict(self):
        return {k: getattr(self, k) for k in dir(self) if
                not k.startswith('_') and k != 'parse' and k != 'state_dict'}