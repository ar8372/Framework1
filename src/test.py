
from collections import defaultdict, Counter


fold_cls_counts = defaultdict(int)
t= [(i,fold_cls_counts[i,8]) for i in range(5)]
print(t)
t= [(i,fold_cls_counts[i,7]) for i in range(5)]
print(t)
t= [(i,fold_cls_counts[i,9]) for i in range(5)]
print(t)