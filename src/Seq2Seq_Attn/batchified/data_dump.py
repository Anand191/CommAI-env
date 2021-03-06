import numpy as np
import pandas as pd
from Seq2Seq_Attn.batchified.lookup_tables_dump import lookup_tables
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
atomic = np.arange(1,7,dtype=int).tolist()
tables2 = [7, 8]
str0, str1, str2 = 'LookupTaskR3D', 'FuncLookupTaskR3D', 'FuncLookupTestTaskR3D'


class data_dump(object):
    def __init__(self, lt_obj):
        self.lt = lt_obj

        self.atomic_dict, self.train_composed, self.heldout_composed, self.test11_subset, self.test12_subset, self.test11_hybrid, self.test12_hybrid, self.test11_unseen,self.test12_unseen = lt_obj.gen_all_data()
        self.data_atomic = np.zeros((len(self.atomic_dict)*8,2),dtype=object)

    def key_rev(self, key):
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

    def all_targets(self, input_sentence):
        ipt = input_sentence.split(' ')
        nis = ipt[0]
        tgt = ipt[0]
        i = 1
        while (i <= len(ipt)):
            if (i == len(ipt)):
                break
            temp = tgt + ' ' + ipt[i]
            row = np.where(self.data_atomic[:, 0] == temp)[0]
            tgt = self.data_atomic[row, 1][0].split(' ')[1] #[row, 1][0]
            nis += ' ' + tgt #ipt[i] + '({})'.format(tgt)
            i += 1
        return nis

    def gen_atomic(self):
        row = 0
        for key,value in self.atomic_dict.items():
            for k, v in self.atomic_dict[key].items():
                self.data_atomic[row,0] = k+' '+key
                self.data_atomic[row,1] = k+ ' '+v
                row += 1

    def dict_to_arr(self, data_dict):
        dim = 0
        for key, values in data_dict.items():
            dim += len(data_dict[key])
        data_com = np.zeros((dim,2),dtype=object)

        row = 0
        for key, value in data_dict.items():
            rev_key = self.key_rev(key)
            for k, v in data_dict[key].items():
                data_com[row, 0] = k + ' ' + rev_key
                data_com[row, 1] = self.all_targets(data_com[row, 0])  # k+ ' '+v
                row += 1

        return data_com

    def plot_data(self,df, name):
        splitter = lambda x: x.split(' ')[-1]
        plt.figure()
        ax = sns.countplot(x=df.iloc[:, -1].apply(splitter))
        ax.set_xlabel('Output Bit Strings')
        plt.savefig("{}.png".format(name))
        plt.close()


    def dump_all(self):
        data_train_val = self.dict_to_arr(self.train_composed)
        np.random.shuffle(data_train_val)
        slice_idx = data_train_val.shape[0] - int(np.round(data_train_val.shape[0] * 0.1))
        data_train, data_val = data_train_val[0:slice_idx,:], data_train_val[slice_idx:,:]
        data_held =self.dict_to_arr(self.heldout_composed)
        data_subset1 = self.dict_to_arr(self.test11_subset)
        data_subset2 = self.dict_to_arr(self.test12_subset)
        data_hybrid1 = self.dict_to_arr(self.test11_hybrid)
        data_hybrid2 = self.dict_to_arr(self.test12_hybrid)
        data_unseen1 = self.dict_to_arr(self.test11_unseen)
        data_unseen2 = self.dict_to_arr(self.test12_unseen)


        master_data_tr = np.vstack((self.data_atomic,data_train))
        master_data_subset = np.vstack((data_subset1, data_subset2))
        master_data_hybrid = np.vstack((data_hybrid1, data_hybrid2))
        master_data_unseen = np.vstack((data_unseen1, data_unseen2))


        df_tr = pd.DataFrame(master_data_tr)
        df_tr.to_csv('./data/train.csv',sep='\t',header=False,index=False)

        df_val = pd.DataFrame(data_val)
        df_val.to_csv('./data/validation.csv',sep='\t',header=False,index=False)

        df_heldout = pd.DataFrame(data_held)
        df_heldout.to_csv('./data/test1_heldout2.csv',sep='\t',header=False,index=False)

        df_subset = pd.DataFrame(master_data_subset)
        df_subset.to_csv('./data/test2_subset2.csv',sep='\t',header=False,index=False)

        df_hybrid = pd.DataFrame(master_data_hybrid)
        df_hybrid.to_csv('./data/test3_hybrid2.csv',sep='\t',header=False,index=False)

        df_unseen = pd.DataFrame(master_data_unseen)
        df_unseen.to_csv('./data/test4_unseen2.csv',sep='\t',header=False,index=False)

        dfs = [df_tr, df_val, df_heldout, df_subset, df_hybrid, df_unseen]
        names = ['train', 'validation', 'held_ipt', 'held_comp', 'held_tab', 'new_comp']
        for i, df in enumerate(dfs):
            self.plot_data(df, names[i])

########################################################################################################################

lookup = lookup_tables(str0, str1, str2, atomic, tables2)
dump = data_dump(lookup)
dump.gen_atomic()
dump.dump_all()

