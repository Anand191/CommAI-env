import os
import random
import argparse
import torch
import numpy as np
import os
#import pandas as pd
from Seq2Seq_Attn.three_bit.Model2 import EncoderRNN, BahdanauAttnDecoderRNN
from Seq2Seq_Attn.three_bit.composed_training import trainIters
from Seq2Seq_Attn.three_bit.evaluate_com import evaluateAndShowAttention
from Seq2Seq_Attn.three_bit.data_com_new import DataPrep, fnames


use_cuda = torch.cuda.is_available()
print("Using Cuda : %s"%use_cuda)

parser = argparse.ArgumentParser()
parser.add_argument('--data', help='Location of Train, Dev and Inference Data', default='./Seq2Seq_Attn/three_bit/data')
parser.add_argument('--infer', help='Location for saving plots', default='./Seq2Seq_Attn/three_bit/Infer_Results')
parser.add_argument('--epochs', type=int, help='Number of epochs', default=100)
parser.add_argument('--embedding_size', type=int, help='Embedding size', default=128)
parser.add_argument('--hidden_size', type=int, help='Hidden layer size', default=256)
parser.add_argument('--print_every', type=int, help='Every how many batches to print results', default=20)
parser.add_argument('--test_every', type=int, help='Every how many batches to print results on test set', default=50)
parser.add_argument('--plot_every', type=int, help='Every how many batches the model should be saved', default=10)
parser.add_argument('--n_layers', type=int, help='Number of RNN layers in both encoder and decoder', default=1)
parser.add_argument('--dropout_p_encoder', type=float, help='Dropout probability for the encoder', default=0.1)
parser.add_argument('--dropout_p_decoder', type=float, help='Dropout probability for the encoder', default=0.1)
parser.add_argument('--lr', type=float, help='Learning rate, recommended settings.\nrecommended settings: adam=0.001 adadelta=1.0 adamax=0.002 rmsprop=0.01 sgd=0.1', default=0.01)
parser.add_argument('--clip', type=float, help='gradient clipping', default=0.25)
parser.add_argument('--use_copy', action='store_true')
parser.add_argument('--use_attn', action='store_true')
parser.add_argument('--use_interim', action='store_true')


opt = parser.parse_args()
print(opt)


class GetData(object):
    def __init__(self, path):
        self.path = path

    def data_prep(self):
        preprocess = DataPrep(self.path, use_cuda)
        preprocess.read_data()
        preprocess.language_pairs()
        preprocess.data_pairs()

        return (preprocess.input_lang, preprocess.output_lang, preprocess.tensor_pairs, preprocess.master_data,
                preprocess.variableFromSentence)


test_accs = np.zeros((5, 2))

preprocessing = GetData(opt.data)
input_lang, output_lang, all_pairs, master_data, vfs = preprocessing.data_prep()

for i in range(1):
    print("*****Starting run {} with {} Epochs*****".format(i,opt.epochs))
    encoder1 = EncoderRNN(input_lang.n_words, opt.hidden_size, opt.embedding_size, use_cuda,n_layers=opt.n_layers,
                          dropout_p=opt.dropout_p_encoder)
    attn_decoder1 = BahdanauAttnDecoderRNN("concat", opt.hidden_size, opt.embedding_size, output_lang.n_words, use_cuda,
                                           n_layers=opt.n_layers, dropout_p=opt.dropout_p_decoder)

    if use_cuda:
        encoder1 = encoder1.cuda()
        attn_decoder1 = attn_decoder1.cuda()

    test_acc = trainIters(encoder1, attn_decoder1, opt.epochs, all_pairs[0], all_pairs[1], all_pairs[2:], fnames[2:], use_cuda,
                          print_every=opt.print_every, plot_every=opt.plot_every, test_every=opt.test_every, learning_rate=opt.lr,
                          use_copy= opt.use_copy, use_attn = opt.use_attn, use_interim=opt.use_interim, clip=opt.clip)
    test_accs[i, 0] = i
    test_accs[i, 1] = test_acc

print('*********End Training*********')
print('')

########################################################################################################################
print('*********Begin Plotting*********')

data_name = ['train', 'validation', 'heldout', 'subset', 'hybrid', 'unseen', 'longer']

for step, data in enumerate(master_data):
    for i in range(0, data.shape[0]):
        ipt_sentence = data[i, 0]
        if (len(ipt_sentence.split(' ')) == 2 and data_name[step] != "train"):
            continue
        else:
            if(data_name[step] == 'longer' and i==60):
                break
            name = os.path.join(opt.infer, data_name[step],'{}{}'.format(data_name[step], i))
            evaluateAndShowAttention(ipt_sentence, encoder1, attn_decoder1, master_data[0], input_lang, output_lang,
                                     use_cuda,vfs, name)


