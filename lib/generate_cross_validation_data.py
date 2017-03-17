#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate training and validation indexes for stratified 10-fold cross-validation
"""
import gzip
import sys
import os
import csv
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

path2data = 'GDI-official-training-data/GDI-TRAIN.txt.gz'
# read in data; each line is a sample \t label
data = []
targets = []
# target encoder
encoder = {'LU':0, 'BE':1, 'ZH':2, 'BS':3}
with gzip.open(path2data) as f:
    for line in f:
        line = line.decode('utf-8')
        d, t = line.split('\t')
        data.append(d)
        # encode targets
        targets.append(encoder[t.strip()])

# first, generate a separate stratified test set
sp = StratifiedShuffleSplit(n_splits=1,
                            test_size=0.1,
                            random_state=42)

train_idx, test_idx = list(sp.split(data, targets))[0]

# second, generate a cross-validation object
cv = StratifiedKFold(n_splits=10, random_state=42,
                     shuffle=True)
# a fold contains approx. 1,450 samples.

# with sklearn.model_selection.GridSearchCV,
# use `cv` as value of `cv` parameter, e.g.:
# gs = GridSearchCV(pipeline, params, n_jobs=-1,cv=cv)

# in a Python script, use generator `cv.split(data, targets)`

splits = cv.split(np.arange(train_idx.size),
                  np.array(targets)[train_idx])

folds = [{'fold' : i,
          'train_idx' : list(train_idx[train]),
          'val_idx' : list(train_idx[val])}
         for i, (train, val) in enumerate(splits)]

# materialize
path = 'cv.d'
if len(sys.argv) > 1:
    path = sys.argv[1]

if not os.path.exists(path):
    os.mkdir(path)
decoder = dict((v, k) for k, v in encoder.items())
rows = [(data[i], decoder[targets[i]]) for i in test_idx]
with open(os.path.join(path, 'test.tsv'), 'w', encoding='utf-8') as w:
    csv.writer(w, delimiter='\t').writerows(rows)

trainrows = [(data[i], decoder[targets[i]]) for i in train_idx]
with open(os.path.join(path, 'train.tsv'), 'w', encoding='utf-8') as w:
    csv.writer(w, delimiter='\t').writerows(trainrows)

for f in folds:
    idx = f['fold']
    for indexes, fn_ in [(f['train_idx'], 'train_%d.tsv'),
                         (f['val_idx'], 'test_%d.tsv')]:
        fn = os.path.join(path, fn_ % idx)
        rows = [(data[i], decoder[targets[i]]) for i in indexes]
        with open(fn, 'w') as w:
            csv.writer(w, delimiter='\t').writerows(rows)

# run some tests on the data to make sure it's fine
try:
    import pandas as pd

    t = pd.DataFrame(data=targets, columns=['target'])
    dist = t.groupby('target').size().reset_index(name='counts')

    ### check test set
    # mean of test indexes should be close to half the data size
    atol = 100
    assert np.isclose(test_idx.mean(), t.size / 2, atol=atol)

    # dist of labels in test set should be same as in the data
    dist_test = t.iloc()[test_idx].groupby(
        'target').size().reset_index(name='counts')

    atol=10
    assert np.allclose(dist.counts, dist_test.counts * 10, atol=atol)

except ImportError:
    pass
def default(o):
    if isinstance(o, np.integer): return int(o)
    raise TypeError
# dump fold indices to stdout

json_filename = "folds.json"

with open(os.path.join(path, json_filename), 'w') as f:
    print(json.dumps({'test_idx' : list(test_idx),
                  'cv_idx' : folds},
                 indent=True,ensure_ascii=True,default=default),file=f)
