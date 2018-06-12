import gensim
import json
from tqdm import tqdm


with open('train_word2vec/corpus_char.txt', 'w') as writer:
    with open('Utils/stopwords.txt')as ff:
        data = ff.readlines()
        stops = [str.strip(i) for i in data]
    for i in tqdm(open('raw/data_train.json', 'r')):
        line = json.loads(i)
        text = line['fact']
        text = ' '.join([i for i in text if i not in stops])
        writer.write(text+'\n')


sentance = gensim.models.word2vec.Text8Corpus('train_word2vec/corpus_char.txt')

model = gensim.models.Word2Vec(sentance, 300)
model.save('train_word2vec/w2v_char.model')

"""

"""
