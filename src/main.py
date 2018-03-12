import os
import argparse
import torch
import numpy as np
import os
import pandas as pd
from Seq2Seq_Attn.Model2 import EncoderRNN, BahdanauAttnDecoderRNN
from Seq2Seq_Attn.modular_reversed.composed_training import trainIters
from Seq2Seq_Attn.modular_reversed.evaluate_com import evaluateRandomly, evaluateAndShowAttention
from Seq2Seq_Attn.modular_reversed.data_com_new import MAX_LENGTH, DataPrep


use_cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser()
parser.add_argument('--data', help='Location of Train, Dev and Inference Data')
parser.add_argument('--infer', help='Location for saving plots')

opt = parser.parse_args()


class GetData(object):
    def __init__(self, path):
        self.path = path

    def data_prep(self):
        preprocess = DataPrep(self.path, use_cuda)
        preprocess.read_data()
        preprocess.language_pairs()
        preprocess.data_pairs()

        return (preprocess.input_lang, preprocess.output_lang, preprocess.training_pairs, preprocess.test_pairs,
                preprocess.master_data,preprocess.variableFromSentence)


test_accs = np.zeros((5, 2))

preprocessing = GetData(opt.data)
input_lang,output_lang,training_pairs,test_pairs,master_data, vfs = preprocessing.data_prep()


for i in range(1):
    print("*****Starting run {}*****".format(i))
    hidden_size = 300
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size, use_cuda)
    attn_decoder1 = BahdanauAttnDecoderRNN("concat", hidden_size, output_lang.n_words, use_cuda,
                                           1, dropout_p=0.1)

    if use_cuda:
        encoder1 = encoder1.cuda()
        attn_decoder1 = attn_decoder1.cuda()

    test_acc = trainIters(encoder1, attn_decoder1, 80, training_pairs, test_pairs, use_cuda, print_every=20,
                          plot_every=10)
    test_accs[i, 0] = i
    test_accs[i, 1] = test_acc

print('beginning_plotting')

data_name = ['train', 'test', 'infer']

for step, data in enumerate(master_data):
    for i in range(0, data.shape[0]):
        ipt_sentence = data[i, 0]
        if (len(ipt_sentence.split(' ')) == 2 and data_name[step] != "train"):
            continue
        else:
            name = os.path.join(opt.infer, data_name[step],'{}{}'.format(data_name[step], i))
            evaluateAndShowAttention(ipt_sentence, encoder1, attn_decoder1, master_data[0], input_lang, output_lang,
                                     use_cuda,vfs, name)


