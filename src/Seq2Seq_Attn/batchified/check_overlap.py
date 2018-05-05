import os
import pandas as pd
import itertools

mfolder = './data'
all_dirs = os.listdir(mfolder)
fnames = ['train.csv', 'validation.csv', 'test1_heldout2.csv']
for dir1 in all_dirs:
    dfs = []
    for f in fnames:
        dfs.append(pd.read_csv(os.path.join(mfolder,dir1,f), delimiter='\t', header=None))
    all_tups = list(itertools.combinations(dfs,2))
    all_names = list(itertools.combinations(fnames,2))
    for i, tup in enumerate(all_tups):
        print(all_names[i][0], all_names[i][1])
        print(tup[0].merge(tup[1]))