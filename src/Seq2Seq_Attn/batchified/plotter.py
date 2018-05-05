import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

sns.set()
mfolder = './Infer_Results'
fnames = ['plot_longer_hardcoded.csv', 'plot_longer_learned.csv', 'plot_longer_baseline.csv']
tnames = ['plot_test_hardcoded.csv', 'plot_test_learned.csv', 'plot_test_baseline.csv']
names1 = ['heldout inputs', 'heldout compositions', 'heldout tables us', 'heldout tables su', 'new compositions']
collate = {'Model':[], 'Trained':[], 'Baseline':[]}

def average(collate2, ncol, names, idx=0, flag=True):
    s = 0
    for key in list(collate2.keys()):
        df = pd.concat(collate2[key], axis=0)
        df = df.groupby(['compName']).mean().reset_index()
        #df.insert(0, 'compName', ncol)
        if flag:
            df.to_csv(os.path.join(mfolder, str(idx) + names[s]), index=False)
        else:
            val = np.repeat(key, len(ncol)).tolist()
            df.insert(0, 'model', val)
            df.to_csv(os.path.join(mfolder, names[s]), index=False)
        s += 1

def update_test(file_n, nfnames, ncol, idx, collate2, flag=False):
    for i in range(1,6):
        mpath = os.path.join(mfolder, 'Run{}'.format(i))
        for j, path in enumerate (list(collate2.keys())):
            temp_names = []
            df = pd.read_csv(os.path.join(mpath, path,file_n), index_col=False)
            lim = int(df.shape[0]/len(ncol))
            for i in range(lim):
                temp_names.extend(ncol)
            df['compName'] = temp_names
            collate2[path].append(df)
    average(collate2, ncol, nfnames, idx)

def plot_variance(file_n):
    collate_plot = {'Model':[], 'Trained':[], 'Baseline':[]}
    for i  in range(1,6):
        mpath = os.path.join(mfolder, 'Run{}'.format(i))
        for j, path in enumerate(list(collate_plot.keys())):
            df = pd.read_csv(os.path.join(mpath, path, file_n), index_col=False)
            collate_plot[path].append(df)

    for key in list(collate_plot.keys()):
        df = pd.concat(collate_plot[key], axis=0)
        errors = df.groupby(['Epoch']).std().reset_index()
        #df = df.groupby(['Epoch']).mean().reset_index()
        fig1 = plt.figure(figsize=(30, 20))
        fig2 = plt.figure(figsize=(30, 20))
        z = 1
        for fd in collate_plot[key]:
            ax = fig1.add_subplot(5,1, z)
            ax.errorbar(fd['Epoch'], fd['Train_Loss'], yerr=1.96*np.asarray(errors['Train_Loss'].values)/np.sqrt(5),fmt = "--o",label='Train')
            ax.errorbar(fd['Epoch'], fd['Test_Loss'], yerr=1.96*np.asarray(errors['Test_Loss'].values)/np.sqrt(5),fmt = "--o",label='Validation')
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Loss")
            ax.legend(loc='best')
            ax.set_title('{} Loss'.format(key))

            ax1 = fig2.add_subplot(5,1, z)
            ax1.errorbar(fd['Epoch'], fd['Train_Acc'], yerr=1.96 * np.asarray(errors['Train_Acc'].values) / np.sqrt(5),
                        fmt="--o", label='Train')
            ax1.errorbar(fd['Epoch'], fd['Test_Acc'], yerr=1.96 * np.asarray(errors['Test_Acc'].values) / np.sqrt(5),
                        fmt="--o", label='Validation')
            ax1.set_xlabel("Epochs")
            ax1.set_ylabel("Accuracy")
            ax1.legend(loc='best')
            ax1.set_title('{} Accuracy'.format(key))

            z += 1
        fig1.tight_layout()
        fig1.savefig(os.path.join('./LC','{}_All_Losses.png'.format(key)))
        plt.close(fig1)
        fig2.tight_layout()
        fig2.savefig(os.path.join('./LC', '{}_All_Accuracies.png'.format(key)))
        plt.close(fig2)
        #plt.show()
plot_variance('plot_data.csv')


# for x in range(3):
#     update_test('plot_test_{}.csv'.format(x), tnames, names1, x, collate, True)
#     for i, key in enumerate(list(collate.keys())):
#         df = pd.read_csv(os.path.join(mfolder, str(x)+tnames[i]), index_col=False)
#         collate[key].append(df)
# average(collate, names1, tnames, flag=False)
#update_longer('plot_longer.csv', fnames, names1[1:])


# print('*********Plotting Learning Curves*************')
# lcs = ['Loss', 'Accuracy']
# axlabels = ['NLL Loss', 'Sequence Accuracy']
# for i in range(1,6):
#     mpath = os.path.join(mfolder, 'Run{}'.format(i))
#     for path in (list(collate.keys())):
#         for j in range(len(lcs)):
#             df = pd.read_csv(os.path.join(mpath,path, 'plot_data.csv'))
#             if(lcs[j]=='Loss'):
#                 col_names = ['Train_Loss', 'Test_Loss']
#             else:
#                 col_names = ['Train_Acc', 'Test_Acc']
#             fig = plt.figure()
#             ax = fig.add_subplot(111)
#             ax.plot(df['Epoch'], df[col_names[0]], label='Train')
#             ax.plot(df['Epoch'], df[col_names[1]], label='Validation')
#             ax.set_xlabel("Epochs")
#             ax.set_ylabel(axlabels[j])
#             ax.legend(loc='best')
#             ax.set_title('%s Curves'%lcs[j])
#             plt.savefig(os.path.join('./LC',path,'Run{}_{}_{}.png'.format(i,path, lcs[j])))
#             plt.close(fig)

#=======================================================================================================================
# f = plt.figure(figsize=(12,10))
# ax = df.groupby(['compLength','compName'])['seqacc'].mean().unstack().plot()
# ax.set_ylabel('Accuracies')
# ax.legend(loc='best')
# # lgd = ax.legend(loc='center', bbox_to_anchor=(0.5,-0.55), ncol=2, fontsize=12)
# # f.savefig('Longer.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
# plt.show()

#=======================================================================================================================
