from Seq2Seq_Attn.lookup_tables_dump3 import atomic_dict, train_composed, val_composed
from Seq2Seq_Attn.lookup_tables_dump3 import test11_unseen, test12_unseen, test21_longer, test22_longer
import numpy as np
import pandas as pd

data_atomic = np.zeros((len(atomic_dict)*4,2),dtype=object)
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

data_train = dict_to_arr(train_composed)
data_val = dict_to_arr(val_composed)
data_unseen1 = dict_to_arr(test11_unseen)
data_unseen2 = dict_to_arr(test12_unseen)
data_longer1 = dict_to_arr(test21_longer)
data_longer2 = dict_to_arr(test22_longer)

master_data_tr = np.vstack((data_atomic,data_train))
data_val, data_held = np.split(data_val,2,axis=0)
master_data_unseen = np.vstack((data_unseen1, data_unseen2))
master_data_longer = np.vstack((data_longer1, data_longer2))

df_tr = pd.DataFrame(master_data_tr)
df_tr.to_csv('train.csv',sep='\t',header=False,index=False)

df_val = pd.DataFrame(data_val)
df_val.to_csv('validation.csv',sep='\t',header=False,index=False)

df_heldout = pd.DataFrame(data_held)
df_heldout.to_csv('heldout.csv',sep='\t',header=False,index=False)

df_unseen = pd.DataFrame(master_data_unseen)
df_unseen.to_csv('unseen.csv',sep='\t',header=False,index=False)

df_longer = pd.DataFrame(master_data_longer)
df_longer.to_csv('longer.csv',sep='\t',header=False,index=False)