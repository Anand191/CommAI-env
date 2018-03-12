import random
from Seq2Seq_Attn.lookup_tables_dump2 import atomic,composed
import numpy as np
import os
import pandas as pd
import torch
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 6

fnames = ['train.csv','test.csv', 'infer.csv']
master_data = []

for fname in fnames:
    df = pd.read_csv(os.path.join('./data', fname), delimiter='\t', header=None)
    master_data.append(df.values)

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS",1: "EOS"} #
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

def readLangs(lang1, lang2, data, reverse=False):
    print("Reading lines...")

    # Split every line into pairs and normalize
    pairs = []
    for i in range(data.shape[0]):
        pairs.append(data[i,:].tolist())


    input_lang = Lang(lang1)
    output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

def prepareData(lang1,lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1,lang2, reverse)
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


input_lang, output_lang, pairs_tr = prepareData('task_tr', 'out_tr',master_data[0])
_, _, pairs_te = prepareData('task_te', 'out_te',master_data[1])
_,_, pairs_if = prepareData('task_if', 'out_if',master_data[2])

# input_lang_te, output_lang_te = input_lang_tr, output_lang_tr
# input_lang_if, output_lang_if = input_lang_tr, output_lang_tr

print(random.choice(pairs_tr))
print(random.choice(pairs_te))
print(random.choice(pairs_if))


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def variableFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result


def variablesFromPair(input_lang, output_lang, pair):
    input_variable = variableFromSentence(input_lang, pair[0])
    target_variable = variableFromSentence(output_lang, pair[1])
    return (input_variable, target_variable)

training_pairs = [variablesFromPair(input_lang, output_lang, pair) for pair in pairs_tr]
test_pairs = [variablesFromPair(input_lang, output_lang, pair) for pair in pairs_te]
infer_pairs = [variablesFromPair(input_lang, output_lang, pair) for pair in pairs_if]