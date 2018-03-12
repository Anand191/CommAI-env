from Seq2Seq_Attn.Model2 import EncoderRNN, BahdanauAttnDecoderRNN
import torch
import numpy as np
import os
import pandas as pd
from Seq2Seq_Attn.reversed_input.composed_training  import trainIters
from Seq2Seq_Attn.reversed_input.evaluate_com import evaluateRandomly,evaluateAndShowAttention
from Seq2Seq_Attn.reversed_input.data_com import input_lang_tr as input_lang,output_lang_tr as output_lang
from Seq2Seq_Attn.reversed_input.data_com import training_pairs, test_pairs, master_data_tr, data_com_te, master_data_if
from Seq2Seq_Attn.reversed_input.data_com import MAX_LENGTH, pairs_tr as pairs1, pairs_te as pairs2, master_data_te
# from Seq2Seq_Attn.reversed_input.data_com import input_lang_te as input_lang2,output_lang_te as output_lang2
# from Seq2Seq_Attn.reversed_input.data_com import input_lang_if as input_lang3, output_lang_if as output_lang3

use_cuda = torch.cuda.is_available()

test_accs = np.zeros((5,2))

for i in range(1):
    print("*****Starting run {}*****".format(i))
    hidden_size = 300
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size, use_cuda)
    attn_decoder1 = BahdanauAttnDecoderRNN("concat", hidden_size, output_lang.n_words, use_cuda,
                                   1, dropout_p=0.1)

    if use_cuda:
        encoder1 = encoder1.cuda()
        attn_decoder1 = attn_decoder1.cuda()

    test_acc = trainIters(encoder1, attn_decoder1, 500, training_pairs,test_pairs, use_cuda, print_every=20, plot_every=10)
    test_accs[i,0] = i
    test_accs[i,1] = test_acc
# df = pd.DataFrame(test_accs,columns=['Experiment No.', 'Final Test Accuracy'])
# df.to_csv("final_test_acc_attn_t5t6.csv")

# print("Evalualte on Training Data")
# evaluateRandomly(encoder1, attn_decoder1, pairs1,input_lang,output_lang, use_cuda)
#
#
# for i in range(master_data_tr.shape[0]):
#     ipt_sentence = master_data_tr[i,0]
#     name = 'train'+'{}'.format(i)
#     evaluateAndShowAttention(ipt_sentence, encoder1, attn_decoder1, master_data_tr, input_lang1, output_lang1, use_cuda,name)
# print('')
#
# print("Evaluate on Test Data")
# evaluateRandomly(encoder1, attn_decoder1, pairs2,input_lang,output_lang, use_cuda)

print('beginning_plotting')

master_data = [master_data_tr,master_data_te,master_data_if]
data_name = ['train','test','infer']

for step, data in enumerate(master_data):
    for i in range(0,data.shape[0]):
        ipt_sentence = data[i,0]
        if(len(ipt_sentence.split(' '))==2 and data_name[step] != "train"):
            continue
        else:
            name = os.path.join('./Infer_Results',data_name[step],'{}{}'.format(data_name[step],i)) #'test'+'{}'.format(i)
            evaluateAndShowAttention(ipt_sentence, encoder1, attn_decoder1, master_data_tr, input_lang, output_lang,
                                     use_cuda,name)

# evaluateAndShowAttention("00 t1 t2",encoder1,attn_decoder1,master_data_te, input_lang, output_lang, use_cuda, "test1")
# evaluateAndShowAttention("11 t2 t3",encoder1,attn_decoder1,master_data_te, input_lang, output_lang, use_cuda, "test2")
#
# evaluateAndShowAttention("01 t3 t1",encoder1,attn_decoder1,master_data_te, input_lang, output_lang, use_cuda, "test3")
# evaluateAndShowAttention("10 t3 t4",encoder1,attn_decoder1,master_data_te, input_lang, output_lang, use_cuda, "test4")

