import random
from Seq2Seq_Attn.lookup_tables_dump import atomic
import numpy as np
import torch
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()
atomic_tasks = atomic()
master_data = np.zeros((len(atomic_tasks)*4,2),dtype=object)
row = 0
for key,value in atomic_tasks.items():
    for k, v in atomic_tasks[key].items():
        master_data[row,0] = key+' '+k
        master_data[row,1] = v+ ' '+k
        row += 1

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 3

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

def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Split every line into pairs and normalize
    pairs = []
    for i in range(master_data.shape[0]):
        pairs.append(master_data[i,:].tolist())


    input_lang = Lang(lang1)
    output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
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


input_lang, output_lang, pairs = prepareData('task', 'out')
print(random.choice(pairs))


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


def variablesFromPair(pair):
    input_variable = variableFromSentence(input_lang, pair[0])
    target_variable = variableFromSentence(output_lang, pair[1])
    return (input_variable, target_variable)

training_pairs = [variablesFromPair(pair) for pair in pairs]




