import json
import jieba
import thulac
from jieba.posseg import POSTokenizer
cut = thulac.thulac(seg_only=True,filt=True )


# with open('raw/data_train.json','r') as f:
#     lines = f.readlines()
#     for i in lines:
#         line = json.loads(i)
#         print(' '.join(jieba.cut(sentence=line['fact'])))
#         print('---')
#         print(' '.join([i[0] for i in cut.cut(line['fact'])]))
#         print(line['meta'])
#         print('=========================')
#         n = input()

with open('raw/data_train.json','r') as f:
    total = 0
    gysh = 0
    one = 0
    two = 0
    three = 0
    four = 0
    five = 0
    six = 0
    seven = 0
    eight = 0
    lines = f.readlines()
    for i in lines:
        line = json.loads(i)
        total+=1
        if '故意伤害' in line['meta']['accusation']:
            gysh+=1
        if len(line['meta']['accusation']) == 1:
            one+=1
        if len(line['meta']['accusation']) == 2:
            two+=2
        if len(line['meta']['accusation']) == 3:
            three+=3
        if len(line['meta']['accusation']) == 4:
            four+=4
        if len(line['meta']['accusation']) == 5:
            five+=5
        if len(line['meta']['accusation']) == 6:
            six+=6
        if len(line['meta']['accusation']) == 7:
            seven+=7
    print(total,gysh)
    print('1',one)
    print('2',two)
    print('3',three)
    print('4',four)
    print('5',five)
    print('6',six)
    print('7',seven)
