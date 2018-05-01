import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

def plot_data(dff, name='img'):
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

def create_df(ipts, lookup):
    holder = np.zeros((len(ipts),2), dtype=object)
    for j, ipt in enumerate(ipts):
        row = np.where(lookup.values[:, 0] == ipt)[0]
        holder[j, 0] = ipt
        holder[j, 1] = lookup.values[row, 1][0]
    return holder


df = pd.read_csv('./data2/composed_train.csv', delimiter='\t')
all_inputs = df['ipt'].values.tolist()
df[['ipt','tab1', 'tab2']] = df['ipt'].str.split(' ', expand=True)
df[['copy','interim', 'opt']] = df['copy'].str.split(' ',expand=True)
df['compositions'] = df[['tab1', 'tab2']].apply(lambda x: ' '.join(x), axis=1)


df2 = pd.read_csv('./data2/balanced.csv', delimiter='\t', index_col=False)
df3 = pd.read_csv('./data2/train.csv', delimiter='\t', header=None)

opts = df2.columns
ipts = opts
inputs = []

for i in range(df2.shape[1]):
    for comp in df2.iloc[:,i].values:
        row = np.where(np.logical_and(df['opt'].values==opts[i-1], df['compositions'].values==comp))[0]
        inputs.append(df['ipt'].iloc[row[0]] + ' ' + comp )

heldout_inputs = list(set(all_inputs) - set(inputs))
arr = create_df(inputs, df3)
brr = create_df(heldout_inputs, df3)

df_com = pd.DataFrame(arr)
df_held = pd.DataFrame(brr)
plot_data(df_com)
plot_data(df_held)

