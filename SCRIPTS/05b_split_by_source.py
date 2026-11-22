r"""
05b_split_by_source.py
Number of records of each data source in training (80 %), internal test (20 %, incl. the buffer
around the re-measured plots) and temporal validation. Reads the split written by 05.

Output  RESULTADOS\00_exploratorio\split_by_source.csv
Run     python 05b_split_by_source.py
"""
import os
import pandas as pd

ROOT = r"D:\PAPER-ECOLIN"
INTERIM = os.path.join(ROOT, "DATA", "interim")
OUT = os.path.join(ROOT, "RESULTADOS", "00_exploratorio")
os.makedirs(OUT, exist_ok=True)

SRC = {'USFQ': 'RBCA', 'BP': 'RBCA', 'RBA': 'RBA', 'GEDI': 'GEDI', 'CAM2': 'Pantropical',
       'SAM3': 'Pantropical', 'SAM4': 'Pantropical', 'SAM5': 'Pantropical'}
ORDER = ['RBCA', 'RBA', 'Pantropical', 'GEDI', 'Weak labels']

d = pd.read_csv(os.path.join(INTERIM, 'split_train_internal.csv'))
d['source'] = d.code.map(SRC).fillna(d.code)
d.loc[d.is_weak_label.astype(bool), 'source'] = 'Weak labels'
print("Values of 'set':", d.set.value_counts().to_dict())

tab = pd.crosstab(d.source, d.set)
if 'in_buffer' in d.columns:                      # test records that are there because of the buffer
    tab['of which buffer'] = d[d.in_buffer.astype(bool)].groupby('source').size()
tv_path = os.path.join(INTERIM, 'split_temporal_validation.csv')
if os.path.exists(tv_path):
    tv = pd.read_csv(tv_path)
    tab['temporal_validation'] = tv.code.map(SRC).fillna(tv.code).value_counts()
tab = tab.reindex([s for s in ORDER if s in tab.index] + [s for s in tab.index if s not in ORDER])
tab = tab.fillna(0).astype(int)
tab.loc['Total'] = tab.sum()
tab.to_csv(os.path.join(OUT, 'split_by_source.csv'))
print("\nRecords by source and set:\n" + tab.to_string())
print(f"\nSaved {os.path.join(OUT, 'split_by_source.csv')}")
