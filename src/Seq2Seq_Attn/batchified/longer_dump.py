import time
from Seq2Seq_Attn.batchified.longer_compositions import longer_splits
import os
import numpy as np
import pandas as pd
import itertools
import random

start_time = time.time()

df = pd.read_csv(os.path.join('./data', 'train.csv'), delimiter='\t', header=None)
atomic_dict = df.values
count = 0
for i in range(atomic_dict.shape[0]):
    ipt = len(atomic_dict[i,0].split(' '))
    if (ipt==2):
        count += 1

data_atomic = atomic_dict[0:count,:]
bin_comb = list(itertools.product([0,1], repeat=3))
bins = [[str(x) for x in tup] for tup in bin_comb]
bin_str = [''.join(bins[i]) for i in range (len(bins))]

def all_targets(input_sentence):
    ipt = input_sentence.split(' ')
    nis = ipt[0]
    tgt = ipt[0]
    i = 1
    while (i <= len(ipt)):
        if (i == len(ipt)):
            break
        temp = tgt + ' ' + ipt[i]
        row = np.where(data_atomic[:, 0] == temp)[0]
        tgt = data_atomic[row, 1][0].split(' ')[1] #[row, 1][0]
        nis += ' ' + tgt #ipt[i] + '({})'.format(tgt)
        i += 1
    return nis

def longer_dump(master,break_p, name):
    if (break_p <= len(bin_str)):
        master_data_longer = np.zeros((len(master)*break_p, 2), dtype=object)
        i = 0
        for slave in master:
            ipt_strs = random.sample(bin_str, break_p)
            key = 't{}'.format(slave[0])
            for j in range(1, len(slave)):
                key += ' t{}'.format(slave[j])
            for ipt in ipt_strs:
                temp_str = ipt + ' ' + key
                target = all_targets(temp_str)
                master_data_longer[i,0] = temp_str
                master_data_longer[i, 1] = target
                i+=1
        df_longer = pd.DataFrame(master_data_longer)
        df_longer.to_csv(os.path.join('./data','test_longer_{}.csv'.format(name)), sep='\t', header=False, index=False)
    else:
        raise ValueError("Maximumum Number of 3bit Strings reached")

splits = ['seen', 'incremental', 'new']
lengths = [50, 50, 7]
comp_length = int(input("Enter Length of Composition:"))

for i, split in enumerate(splits):
    obj = longer_splits(comp_length,split,lengths[i], 100)
    splits = obj.all_composition()
    #print(obj.max_size)
    longer_dump(list(set(splits)),3, split+str(comp_length))
print("done")
print("Main Program Run:--- %s seconds ---" % (time.time() - start_time))