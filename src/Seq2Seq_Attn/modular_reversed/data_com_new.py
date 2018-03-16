import random
import torch
from torch.autograd import Variable
import pandas as pd
import os

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 6

fnames = ['train.csv','validation.csv', 'heldout.csv', 'unseen.csv', 'longer.csv']

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
        self.input_lang, self.output_lang, self.pairs_tr = self.prepareData('task_tr', 'out_tr',self.master_data[0])
        _, _, self.pairs_te = self.prepareData('task_te', 'out_te',self.master_data[1])
        _, _, self.pairs_if1 = self.prepareData('task_if1', 'out_if1',self.master_data[2])
        _, _, self.pairs_if2 = self.prepareData('task_if2', 'out_if2', self.master_data[3])
        _, _, self.pairs_if3 = self.prepareData('task_if3', 'out_if3', self.master_data[4])

        # input_lang_te, output_lang_te = input_lang_tr, output_lang_tr
        # input_lang_if, output_lang_if = input_lang_tr, output_lang_tr

        print(random.choice(self.pairs_tr))
        print(random.choice(self.pairs_te))
        print(random.choice(self.pairs_if1))
        print(random.choice(self.pairs_if2))
        print(random.choice(self.pairs_if3))


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
        self.training_pairs = [self.variablesFromPair(self.input_lang, self.output_lang, pair) for pair in self.pairs_tr]
        self.test_pairs = [self.variablesFromPair(self.input_lang, self.output_lang, pair) for pair in self.pairs_te]
        self.infer_pairs1 = [self.variablesFromPair(self.input_lang, self.output_lang, pair) for pair in self.pairs_if1]
        self.infer_pairs2 = [self.variablesFromPair(self.input_lang, self.output_lang, pair) for pair in self.pairs_if2]
        self.infer_pairs3 = [self.variablesFromPair(self.input_lang, self.output_lang, pair) for pair in self.pairs_if3]