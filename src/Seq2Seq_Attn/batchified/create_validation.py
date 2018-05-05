import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

class uniform_valid(object):
    def __init__(self):
        self.df0 = pd.read_csv('./data/init_test1_heldout2.csv', delimiter='\t', header=None)
        tarr = self.df0.values
        self.df = pd.DataFrame(tarr, columns=['ipt', 'copy'])
        self.all_inputs = self.df['ipt'].values.tolist()
        self.df[['ipt', 'tab1', 'tab2']] = self.df['ipt'].str.split(' ', expand=True)
        self.df[['copy', 'interim', 'opt']] = self.df['copy'].str.split(' ', expand=True)
        self.df['compositions'] = self.df[['tab1', 'tab2']].apply(lambda x: ' '.join(x), axis=1)
        self.df3 = pd.read_csv('./data/init_test1_heldout2.csv', delimiter='\t', header=None)

    def play_sudoku(self):
        arr = np.zeros((5, 8))
        Y = self.df['compositions'].unique().tolist()
        Ybucket = {}
        for y in Y:
            Ybucket[y] = 0
        sudoku = pd.DataFrame(arr, columns=self.df['opt'].unique().tolist())
        count = 0
        total = 40
        for j in range(0, sudoku.shape[0]):
            for opt in sudoku.columns:
                compositions = [x for x in list(Ybucket.keys()) if Ybucket[x] <= 2]
                #print(compositions)
                while True:
                    c = np.random.choice(compositions, 1)
                    if (c[0] not in sudoku[opt].iloc[:].values):
                        row = np.where(np.logical_and(self.df['opt'].values == opt,
                                                      self.df['compositions'].values == c[0]))[0]
                        #temp_ipt = self.df['ipt'].iloc[row[0]] + ' ' + c[0]
                        if len(row) != 0:
                            break
                    else:
                        continue
                count += 1
                left = total - count
                print("{} found -- {} left".format(count, left))
                sudoku[opt].iloc[j] = c[0]
                Ybucket[c[0]] += 1
        #print(sudoku)
        self.df2 = sudoku


    def plot_data(self, dff, name='img'):
        splitter = lambda x: x.split(' ')[-1]
        splitter2 = lambda x: ' '.join(map(str, x.split(' ')[1:]))

        plt.figure()
        ax1 = sns.countplot(x=dff.iloc[:, -1].apply(splitter))
        ax1.set_xlabel('Output Bit Strings')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
        plt.savefig('./data/{}{}.png'.format(name,1))
        plt.close()
        #plt.show()

        plt.figure()
        ax2 = sns.countplot(x=dff.iloc[:, 0].apply(splitter2))
        ax2.set_xlabel('Compositions')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)
        plt.savefig('./data/{}{}.png'.format(name, 2))
        plt.close()
        #plt.show()

    def create_df(self, ipts):
        holder = np.zeros((len(ipts),2), dtype=object)
        for j, ipt in enumerate(ipts):
            row = np.where(self.df3.values[:, 0] == ipt)[0]
            holder[j, 0] = ipt
            holder[j, 1] = self.df3.values[row, 1][0]
        return holder

    def create_uniform(self):
        opts = self.df2.columns
        print(len(opts))
        inputs = []
        for i in range(self.df2.shape[1]):
            for comp in self.df2.iloc[:,i].values:
                row = np.where(np.logical_and(self.df['opt'].values==opts[i], self.df['compositions'].values==comp))[0]
                inputs.append(self.df['ipt'].iloc[row[0]] + ' ' + comp )
        heldout_inputs = list(set(self.all_inputs) - set(inputs))
        arr = self.create_df(inputs)
        brr = self.create_df(heldout_inputs)
        df_com = pd.DataFrame(arr)
        df_held = pd.DataFrame(brr)
        self.plot_data(df_com, 'held_ipt')
        self.plot_data(df_held, 'valid')

        return (df_com, df_held)

val_unif = uniform_valid()
val_unif.play_sudoku()
df_held, df_val = val_unif.create_uniform()
df_held.to_csv('./data/test1_heldout2.csv',sep='\t',header=False,index=False)
df_val.to_csv('./data/validation.csv',sep='\t',header=False,index=False)

