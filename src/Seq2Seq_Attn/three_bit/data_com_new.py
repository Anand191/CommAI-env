import random
import torch
from torch.autograd import Variable
import pandas as pd
import os

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 6

fnames = ['train.csv','validation.csv', 'test1_heldout.csv','test2_subset.csv', 'test3_hybrid.csv',
          'test4_unseen.csv', 'test5_longer.csv']

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


class DataPrep(object):
    def __init__(self,path, use_cuda):
        self.path = path
        self.use_cuda = use_cuda
        self.master_data = []
        self.pairs = []
        self.tensor_pairs = []

    def read_data(self):
        for fname in fnames:
            df = pd.read_csv(os.path.join(self.path, fname), delimiter='\t', header=None)
            self.master_data.append(df.values)

    def readLangs(self,lang1, lang2, data, reverse=False):
        print("Reading lines...")

        # Split every line into pairs and normalize
        pairs = []
        for i in range(data.shape[0]):
            pairs.append(data[i,:].tolist())
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

        return input_lang, output_lang, pairs

    def prepareData(self,lang1,lang2, reverse=False):
        input_lang, output_lang, pairs = self.readLangs(lang1,lang2, reverse)
        print("Read %s sentence pairs" % len(pairs))
        print("Trimmed to %s sentence pairs" % len(pairs))
        print("Counting words...")
        for pair in pairs:
            input_lang.addSentence(pair[0])
            output_lang.addSentence(pair[1])
        print("Counted words:")
        print(input_lang.name, input_lang.n_words)
        print(output_lang.name, output_lang.n_words)
        return input_lang, output_lang, pairs

    def language_pairs(self):
        self.input_lang, self.output_lang, pairs_tr = self.prepareData('task_tr', 'out_tr',self.master_data[0])
        self.pairs.append(pairs_tr)
        for i in range(1, len(self.master_data)):
            _,_,pairs_temp = self.prepareData('task','out', self.master_data[i])
            self.pairs.append(pairs_temp)

        for pair in self.pairs:
            print(random.choice(pair))


    def indexesFromSentence(self,lang, sentence):
        return [lang.word2index[word] for word in sentence.split(' ')]


    def variableFromSentence(self,lang, sentence):
        indexes = self.indexesFromSentence(lang, sentence)
        indexes.append(EOS_token)
        result = Variable(torch.LongTensor(indexes).view(-1, 1))
        if self.use_cuda:
            return result.cuda()
        else:
            return result

    def variablesFromPair(self,input_lang, output_lang, pair):
        input_variable = self.variableFromSentence(input_lang, pair[0])
        target_variable = self.variableFromSentence(output_lang, pair[1])
        return (input_variable, target_variable)

    def data_pairs(self):
        for i in range(len(self.pairs)):
            self.data_pair = [self.variablesFromPair(self.input_lang, self.output_lang, pair) for pair in self.pairs[i]]
            self.tensor_pairs.append(self.data_pair)