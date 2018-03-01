import random
from Seq2Seq_Attn.lookup_tables_dump2 import atomic,composed
import numpy as np
import torch
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()
atomic_tasks = atomic()
composed_tasks = composed()
composed_train, composed_test,composed_infer = composed_tasks[0], composed_tasks[1], composed_tasks[2]

data_atomic = np.zeros((len(atomic_tasks)*4,2),dtype=object)
row = 0

def key_rev(key):
    key_elem = key.split(' ')
    i = -1
    rev_key = ''
    while(i>=-len(key_elem)):
        if (i == -len(key_elem)):
            rev_key += key_elem[i]
            break
        rev_key += key_elem[i]+' '
        i -= 1
    return rev_key


for key,value in atomic_tasks.items():
    for k, v in atomic_tasks[key].items():
        data_atomic[row,0] = k+' '+key
        data_atomic[row,1] = k+ ' '+v
        row += 1

dim_tr,dim_te, dim_if = 0, 0, 0
for key,values in composed_train.items():
    dim_tr += len(composed_train[key])

for key,values in composed_test.items():
    dim_te += len(composed_test[key])

for key,values in composed_infer.items():
    dim_if += len(composed_infer[key])

data_com_tr = np.zeros((dim_tr,2),dtype=object)
data_com_te = np.zeros((dim_te,2),dtype=object)
data_com_if = np.zeros((dim_if,2),dtype=object)
row = 0
for key,value in composed_train.items():
    rev_key = key_rev(key)
    for k, v in composed_train[key].items():
        data_com_tr[row,0] = k+' '+rev_key
        data_com_tr[row,1] = k+ ' '+v
        row += 1
row = 0
for key,value in composed_test.items():
    rev_key = key_rev(key)
    for k, v in composed_test[key].items():
        data_com_te[row,0] = k+' '+rev_key
        data_com_te[row,1] = k+ ' '+v
        row += 1

row = 0
for key,value in composed_infer.items():
    rev_key = key_rev(key)
    for k, v in composed_infer[key].items():
        data_com_if[row,0] = k+' '+rev_key
        data_com_if[row,1] = k+ ' '+v
        row += 1

master_data_tr = np.vstack((data_atomic,data_com_tr))
master_data_te = np.vstack((data_atomic,data_com_te))
master_data_if = np.vstack((data_atomic,data_com_if))

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 6

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


input_lang_tr, output_lang_tr, pairs_tr = prepareData('task_tr', 'out_tr',master_data_tr)
input_lang_te, output_lang_te, pairs_te = prepareData('task_te', 'out_te',data_com_te)
input_lang_if, output_lang_if, pairs_if = prepareData('task_if', 'out_if',data_com_if)

input_lang_te, output_lang_te = input_lang_tr, output_lang_tr
input_lang_if, output_lang_if = input_lang_tr, output_lang_tr

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

training_pairs = [variablesFromPair(input_lang_tr, output_lang_tr, pair) for pair in pairs_tr]
test_pairs = [variablesFromPair(input_lang_te, output_lang_te, pair) for pair in pairs_te]
infer_pairs = [variablesFromPair(input_lang_if, output_lang_if, pair) for pair in pairs_if]