import gensim
import os
import json
from tqdm import tqdm

im = []
for i in tqdm(os.listdir('processed/data_train/')):
    with open('processed/data_train/'+i, 'r') as f:
        line = json.loads(f.readline())
        im.append(line['label']['accusation'])


bb = [sum(i) for i in im]
max(bb)