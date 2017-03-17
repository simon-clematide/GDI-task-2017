#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compute mean and stddev from wapiti
"""

import re
import sys

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())
sys.stdin = codecs.getreader("utf-8")(sys.stdin.detach())

import numpy as np

data = []
accuracy = []
for a in sys.argv[1:]:
    if not a.endswith('err'):
        continue
    print('#reading', a, file=sys.stderr)
    with open(a, 'r', encoding='utf-8') as f:
        for l in f:
            #     Sequence error: 12.12%
            if 'Sequence error' in l:
                m = re.search(r'(\d+\.\d+)%', l)
                if m:
                    data.append(float(m.group(1)))
                    accuracy.append(100 - float(m.group(1)))
print(data, file=sys.stderr)
print(np.mean(accuracy), np.std(accuracy), np.mean(data), sep="\t")
