from Seq2Seq_Attn.batchified.lookup_tables_dump import atomic_dict, train_composed, heldout_composed
from Seq2Seq_Attn.batchified.lookup_tables_dump import test11_subset, test12_subset, test11_hybrid, test12_hybrid
from Seq2Seq_Attn.batchified.lookup_tables_dump import test11_unseen, test12_unseen, test11_longer, test12_longer
import numpy as np
import pandas as pd

data_atomic = np.zeros((len(atomic_dict)*8,2),dtype=object)
row = 0

def key_rev(key):
    key_elem = key.split(' ')
    i = -1
    rev_key = ''
    while(i>=-len(key_elem)):
        if (i == -len(key_elem)):
            rev_key += key_elem[i]
            break
        rev_key += key_elem[i]+' '
        i -= 1
    return rev_key

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
        tgt = data_atomic[row, 1][0].split(' ')[1]
        nis += ' ' + tgt #ipt[i] + '({})'.format(tgt)
        i += 1
    return nis

for key,value in atomic_dict.items():
    for k, v in atomic_dict[key].items():
        data_atomic[row,0] = k+' '+key
        data_atomic[row,1] = k+ ' '+v
        row += 1

def dict_to_arr(data_dict):
    dim = 0
    for key, values in data_dict.items():
        dim += len(data_dict[key])
    data_com = np.zeros((dim,2),dtype=object)

    row = 0
    for key, value in data_dict.items():
        rev_key = key_rev(key)
        for k, v in data_dict[key].items():
            data_com[row, 0] = k + ' ' + rev_key
            data_com[row, 1] = all_targets(data_com[row, 0])  # k+ ' '+v
            row += 1

    return data_com

data_train_val = dict_to_arr(train_composed)
np.random.shuffle(data_train_val)
slice_idx = data_train_val.shape[0] - int(np.round(data_train_val.shape[0] * 0.1))
data_train, data_val = data_train_val[0:slice_idx,:], data_train_val[slice_idx:,:]
data_held = dict_to_arr(heldout_composed)
data_subset1 = dict_to_arr(test11_subset)
data_subset2 = dict_to_arr(test12_subset)
data_hybrid1 = dict_to_arr(test11_hybrid)
data_hybrid2 = dict_to_arr(test12_hybrid)
data_unseen1 = dict_to_arr(test11_unseen)
data_unseen2 = dict_to_arr(test12_unseen)
data_longer1 = dict_to_arr(test11_longer)
data_longer2 = dict_to_arr(test12_longer)

master_data_tr = np.vstack((data_atomic,data_train))
master_data_subset = np.vstack((data_subset1, data_subset2))
master_data_hybrid = np.vstack((data_hybrid1, data_hybrid2))
master_data_unseen = np.vstack((data_unseen1, data_unseen2))
master_data_longer = np.vstack((data_longer1, data_longer2))

df_tr = pd.DataFrame(master_data_tr)
df_tr.to_csv('./data/train.csv',sep='\t',header=False,index=False)

df_val = pd.DataFrame(data_val)
df_val.to_csv('./data/validation.csv',sep='\t',header=False,index=False)

df_heldout = pd.DataFrame(data_held)
df_heldout.to_csv('./data/test1_heldout.csv',sep='\t',header=False,index=False)

df_subset = pd.DataFrame(master_data_subset)
df_subset.to_csv('./data/test2_subset.csv',sep='\t',header=False,index=False)

df_hybrid = pd.DataFrame(master_data_hybrid)
df_hybrid.to_csv('./data/test3_hybrid.csv',sep='\t',header=False,index=False)

df_unseen = pd.DataFrame(master_data_unseen)
df_unseen.to_csv('./data/test4_unseen.csv',sep='\t',header=False,index=False)

df_longer = pd.DataFrame(master_data_longer)
df_longer.to_csv('./data/test5_longer.csv',sep='\t',header=False,index=False)