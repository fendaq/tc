import json
import jieba


with open('Utils/stopwords.txt')as ff:
    data = ff.readlines()
    stops = [str.strip(i) for i in data]

with open('raw/data_train.json') as f:
    with open('Utils/stopwords.txt')as ff:
        data = ff.readlines()
        stops = [str.strip(i) for i in data]
    data = f.readlines()
    for i in data:
        line = json.loads(i)
        fact = line['fact']
        cutted = list(jieba.cut(fact))
        droped = [i for i in cutted if i not in stops]
        print('..',len(cutted),len(droped))