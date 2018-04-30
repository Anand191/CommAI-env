import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

sns.set()
#fpaths = ['./Infer_Results/Model', './Infer_Results/Trained','./Infer_Results/Baseline']
#fpaths = ['./Hyperparam Search/1000_1000_SGD_0.001/Model', './Hyperparam Search/1000_1000_SGD_0.001/Baseline']
fpaths = []
folder_name = input("Name of Data Folder:")
sub_folders = ['Model','Baseline']
for sub in sub_folders:
    fpaths.append(os.path.join('./Hyperparam Search',folder_name,sub))
fnames = ['plot_longer_hardcoded.csv', 'plot_longer_learned.csv', 'plot_longer_baseline.csv']
tnames = ['plot_test_hardcoded.csv', 'plot_test_learned.csv', 'plot_test_baseline.csv']

names1 = ['heldout inputs', 'heldout compositions', 'heldout tables', 'new compositions']

# def update(file_n, nfnames, ncol, flag=False):
#     for j, path in enumerate (fpaths):
#         temp_names = []
#         df = pd.read_csv(os.path.join(path,file_n), index_col=False)
#         lim = int(df.shape[0]/len(ncol))
#         for i in range(lim):
#             temp_names.extend(ncol)
#         df['compName'] = temp_names
#
#         if flag:
#             val = np.repeat(path.split('/')[-1], len(ncol)).tolist()
#             df.insert(0,'model', val)
#         df.to_csv(os.path.join(path, nfnames[j]), index=False)
# update('plot_test.csv', tnames, names1, True)
# update('plot_longer.csv', fnames, names1[1:])


# f = plt.figure(figsize=(12,10))
# ax = df.groupby(['compLength','compName'])['seqacc'].mean().unstack().plot()
# ax.set_ylabel('Accuracies')
# ax.legend(loc='best')
# # lgd = ax.legend(loc='center', bbox_to_anchor=(0.5,-0.55), ncol=2, fontsize=12)
# # f.savefig('Longer.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
# plt.show()

print('*********Plotting Learning Curves*************')
lcs = ['Loss', 'Accuracy']
axlabels = ['NLL Loss', 'Sequence Accuracy']

for path in fpaths:
    for i in range(len(lcs)):
        df = pd.read_csv(os.path.join(path,'plot_data.csv'))
        if(lcs[i]=='Loss'):
            col_names = ['Train_Loss', 'Test_Loss']
        else:
            col_names = ['Train_Acc', 'Test_Acc']
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(df['Epoch'], df[col_names[0]], label='Train')
        ax.plot(df['Epoch'], df[col_names[1]], label='Validation')
        ax.set_xlabel("Epochs")
        ax.set_ylabel(axlabels[i])
        ax.legend(loc='best')
        ax.set_title('%s Curves'%lcs[i])
        plt.savefig(os.path.join(path,'{}.png'.format(lcs[i])))
        plt.close(fig)