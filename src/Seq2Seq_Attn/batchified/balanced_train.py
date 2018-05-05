import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

class uniform_train(object):
    def __init__(self, df):
        self.df = pd.read_csv('./data/composed_train.csv', delimiter='\t')
        self.all_inputs = self.df['ipt'].values.tolist()
        self.df[['ipt', 'tab1', 'tab2']] = self.df['ipt'].str.split(' ', expand=True)
        self.df[['copy', 'interim', 'opt']] = self.df['copy'].str.split(' ', expand=True)
        self.df['compositions'] = self.df[['tab1', 'tab2']].apply(lambda x: ' '.join(x), axis=1)
        self.df2 = pd.read_csv('./data/balanced.csv', delimiter='\t', index_col=False)
        self.df3 = df #pd.read_csv('./data2/train.csv', delimiter='\t', header=None)

    def plot_data(self, dff, name='img'):
        splitter = lambda x: x.split(' ')[-1]
        splitter2 = lambda x: ' '.join(map(str, x.split(' ')[1:]))

        plt.figure()
        ax1 = sns.countplot(x=dff.iloc[:, -1].apply(splitter))
        ax1.set_xlabel('Output Bit Strings')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
        plt.show()

        plt.figure()
        ax2 = sns.countplot(x=dff.iloc[:, 0].apply(splitter2))
        ax2.set_xlabel('Compositions')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)

        plt.show()

    def create_df(self, ipts):
        holder = np.zeros((len(ipts),2), dtype=object)
        for j, ipt in enumerate(ipts):
            row = np.where(self.df3.values[:, 0] == ipt)[0]
            holder[j, 0] = ipt
            holder[j, 1] = self.df3.values[row, 1][0]
        return holder

    def create_uniform(self):
        opts = self.df2.columns
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
        self.plot_data(df_com)
        self.plot_data(df_held)

        return (df_com, df_held)

