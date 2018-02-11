import random
from Seq2Seq_Attn.lookup_tables_dump import atomic,composed
import numpy as np
import torch
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()
atomic_tasks = atomic()
composed_tasks = composed()
composed_train, composed_test = composed_tasks[0], composed_tasks[1]

data_atomic = np.zeros((len(atomic_tasks)*4,2),dtype=object)
row = 0
for key,value in atomic_tasks.items():
    for k, v in atomic_tasks[key].items():
        data_atomic[row,0] = key+' '+k
        data_atomic[row,1] = v+ ' '+k
        row += 1

data_com_tr = np.zeros((len(composed_train)*2,2),dtype=object)
data_com_te = np.zeros((len(composed_test)*2,2),dtype=object)
row = 0
for key,value in composed_train.items():
    for k, v in composed_train[key].items():
        data_com_tr[row,0] = key+' '+k
        data_com_tr[row,1] = v+ ' '+k
        row += 1
row = 0
for key,value in composed_test.items():
    for k, v in composed_test[key].items():
        data_com_te[row,0] = key+' '+k
        data_com_te[row,1] = v+ ' '+k
        row += 1

master_data_tr = np.vstack((data_atomic,data_com_tr))

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 4

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

def readLangs(lang1, lang2, data, reverse=False):
    print("Reading lines...")

    # Split every line into pairs and normalize
    pairs = []
    for i in range(data.shape[0]):
        pairs.append(data[i,:].tolist())


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


input_lang_tr, output_lang_tr, pairs_tr = prepareData('task_tr', 'out_tr',master_data_tr)
input_lang_te, output_lang_te, pairs_te = prepareData('task_te', 'out_te',data_com_te)
print(random.choice(pairs_tr))
print(random.choice(pairs_te))


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

training_pairs = [variablesFromPair(input_lang_tr, output_lang_tr, pair) for pair in pairs_tr]
test_pairs = [variablesFromPair(input_lang_te, output_lang_te, pair) for pair in pairs_te]