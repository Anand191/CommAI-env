import os
import random
import argparse
import torch
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from Seq2Seq_Attn.batchified.evaluate_com import evaluateAndShowAttention
from Seq2Seq_Attn.batchified.data_com_new import DataPrep, long_names
from Seq2Seq_Attn.batchified.infer_com import inferIters
from Seq2Seq_Attn.batchified.checkpoint import checkpoint

use_cuda = torch.cuda.is_available()
print("Using Cuda : %s"%use_cuda)

parser = argparse.ArgumentParser()
parser.add_argument('--data', help='Location of Train, Dev and Inference Data', default='./Seq2Seq_Attn/batchified/data')
parser.add_argument('--infer', help='Location for saving plots', default='./Seq2Seq_Attn/batchified/Infer_Results')
parser.add_argument('--encoder_weights', help='Location for saving encoder weights', default='./Seq2Seq_Attn/batchified/Encoder')
parser.add_argument('--decoder_weights', help='Location for saving decoder weights', default='./Seq2Seq_Attn/batchified/Decoder')
parser.add_argument('--use_copy', action='store_true')
parser.add_argument('--use_attn', action='store_true')
parser.add_argument('--use_interim', action='store_true')
parser.add_argument('--train_attn', action='store_true')
parser.add_argument('--max_comp_len', type=int, help='length of longest composition', default=10)


opt = parser.parse_args()
print(opt)

# print('*********Plotting Learning Curves*************')
# lcs = ['Loss', 'Accuracy']
# for i in range(len(lcs)):
#     df = pd.read_csv(os.path.join(opt.infer,'plot_data.csv'))
#     if(lcs[i]=='Loss'):
#         col_names = ['Train_Loss', 'Test_Loss']
#     else:
#         col_names = ['Train_Acc', 'Test_Acc']
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.plot(df['Epoch'], df[col_names[0]], label='Train')
#     ax.plot(df['Epoch'], df[col_names[1]], label='Validation')
#     ax.set_xlabel("Epochs")
#     ax.set_ylabel("NLL Loss")
#     ax.legend(loc='best')
#     ax.set_title('%s Curves'%lcs[i])
#     plt.savefig(os.path.join(opt.infer,'{}.png'.format(lcs[i])))
#     plt.close(fig)

print('*********Run in Inference Mode**********')
folders = ['Test1', 'Test2', 'Test3']
fnames = ['train.csv','validation.csv', 'test1_heldout2.csv','test2_subset2.csv', 'hybrid_UnseenSeen.csv',
          'hybrid_SeenUnseen.csv','test4_unseen2.csv'] #'test3_hybrid2.csv',
split_names = ['seen', 'incremental', 'new']
longer_names = long_names(opt.max_comp_len)
tnames = fnames[2:]
lnames = longer_names.get_lnames(split_names)
fnames = fnames+lnames

path1 = opt.encoder_weights
path2 = opt.decoder_weights
cp1 = checkpoint.load(path1,use_cuda)
cp2 = checkpoint.load(path2,use_cuda)
encoder = cp1.model
decoder = cp2.model
input_vocab = cp1.input_vocab
output_vocab = cp2.output_vocab
# print(input_vocab.index2word)
# print(output_vocab.index2word)
# input()

class GetData(object):
    def __init__(self, path):
        self.path = path

    def data_prep(self, inlang=None, outlang=None):
        preprocess = DataPrep(self.path, use_cuda)
        preprocess.read_data(fnames)
        preprocess.language_pairs(inlang, outlang)
        preprocess.data_pairs()
        # print(preprocess.input_lang.index2word)
        # print(preprocess.output_lang.index2word)
        # input()

        return (preprocess.tensor_pairs, preprocess.master_data,preprocess.variableFromSentence)

for x, fd in enumerate(folders):
    preprocessing = GetData(os.path.join(opt.data,fd))
    all_pairs, master_data, vfs = preprocessing.data_prep(input_vocab, output_vocab)
    plot_data = np.zeros((len(lnames), 5), dtype=object)
    plot_test = np.zeros((len(tnames),4), dtype=object)
    infer_pairs = all_pairs[2:]
    comp_len = []
    for n in range(3,opt.max_comp_len+1):
        arr = np.repeat(n,3).tolist()
        comp_len += arr
    plot_data[:,0] = np.asarray(comp_len)

    j = 0
    k = 0
    for i, ip in enumerate(infer_pairs):
        res = inferIters(encoder, decoder, ip, use_cuda, opt.use_copy, opt.use_attn, opt.use_interim, opt.train_attn, fnames[2:][i])
        if(fnames[2:][i] not in lnames):
            plot_test[k,:] = np.asarray(list(res))
            k += 1
        else:
            #continue
            plot_data[j,1:] = np.asarray(list(res))
            j += 1
        #print(plot_test)
    fd = pd.DataFrame(plot_test, columns=['compName', 'wordacc', 'seqacc', 'finaltargetacc'])
    fd.to_csv(os.path.join(opt.infer,'plot_test_{}.csv'.format(x)), index=False)
    df = pd.DataFrame(plot_data, columns=['compLength', 'compName', 'wordacc', 'seqacc', 'finaltargetacc'])
    df.to_csv(os.path.join(opt.infer,'plot_longer_{}.csv'.format(x)), index=False)

########################################################################################################################
# print('*********Begin Plotting*********')
# preprocessing = GetData(os.path.join(opt.data,'Test1'))
# all_pairs, master_data, vfs = preprocessing.data_prep(input_vocab, output_vocab)
#
# data_name = ['train', 'validation', 'hld_ipt', 'hld_comp', 'hld_tab_us', 'hld_tab_su', 'new_comp', 'hld_comp_3',
#              'hld_tab_3', 'new_comp_3']
#
# for step, data in enumerate(master_data):
#     if(data_name[step].split('_')[-1] == '3'):
#         np.random.shuffle(data)
#     for i in range(0, data.shape[0]):
#         ipt_sentence = data[i, 0]
#         if (len(ipt_sentence.split(' ')) == 2 and data_name[step] != "train"):
#             continue
#         else:
#             if(data_name[step].split('_')[-1] == '3' and i==100):
#                 break
#             name = os.path.join(opt.infer, 'AttentionPlots', data_name[step],'{}{}'.format(data_name[step], i))
#             evaluateAndShowAttention(ipt_sentence, encoder, decoder, master_data[0], input_vocab, output_vocab,
#                                      use_cuda,vfs, name, opt.train_attn)


