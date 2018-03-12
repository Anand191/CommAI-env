import random
from Seq2Seq_Attn.lookup_tables_dump2 import atomic,composed
import numpy as np
import pandas as pd

atomic_tasks = atomic()
composed_tasks = composed()
composed_train, composed_test,composed_infer = composed_tasks[0], composed_tasks[1], composed_tasks[2]

data_atomic = np.zeros((len(atomic_tasks)*4,2),dtype=object)
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


for key,value in atomic_tasks.items():
    for k, v in atomic_tasks[key].items():
        data_atomic[row,0] = k+' '+key
        data_atomic[row,1] = k+ ' '+v
        row += 1

dim_tr,dim_te, dim_if = 0, 0, 0
for key,values in composed_train.items():
    dim_tr += len(composed_train[key])

for key,values in composed_test.items():
    dim_te += len(composed_test[key])

for key,values in composed_infer.items():
    dim_if += len(composed_infer[key])

data_com_tr = np.zeros((dim_tr,2),dtype=object)
data_com_te = np.zeros((dim_te,2),dtype=object)
data_com_if = np.zeros((dim_if,2),dtype=object)
row = 0
for key,value in composed_train.items():
    rev_key = key_rev(key)
    for k, v in composed_train[key].items():
        data_com_tr[row,0] = k+' '+rev_key
        data_com_tr[row,1] = k+ ' '+v
        row += 1
row = 0
for key,value in composed_test.items():
    rev_key = key_rev(key)
    for k, v in composed_test[key].items():
        data_com_te[row,0] = k+' '+rev_key
        data_com_te[row,1] = k+ ' '+v
        row += 1

row = 0
for key,value in composed_infer.items():
    rev_key = key_rev(key)
    for k, v in composed_infer[key].items():
        data_com_if[row,0] = k+' '+rev_key
        data_com_if[row,1] = k+ ' '+v
        row += 1

master_data_tr = np.vstack((data_atomic,data_com_tr))

df_tr = pd.DataFrame(master_data_tr)
df_te = pd.DataFrame(data_com_te)
df_if = pd.DataFrame(data_com_if)

df_tr.to_csv('./data/train.csv',sep='\t',header=False,index=False)
df_te.to_csv('./data/test.csv',sep='\t',header=False,index=False)
df_if.to_csv('./data/infer.csv',sep='\t',header=False,index=False)