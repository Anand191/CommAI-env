from Seq2Seq_Attn.Model2 import EncoderRNN, BahdanauAttnDecoderRNN
import torch
import numpy as np
import pandas as pd
from Seq2Seq_Attn.reversed_input.composed_training  import trainIters
from Seq2Seq_Attn.reversed_input.evaluate_com import evaluateRandomly,evaluateAndShowAttention
from Seq2Seq_Attn.reversed_input.data_com import input_lang_tr as input_lang1,output_lang_tr as output_lang1
from Seq2Seq_Attn.reversed_input.data_com import input_lang_te as input_lang2,output_lang_te as output_lang2
from Seq2Seq_Attn.reversed_input.data_com import training_pairs, test_pairs, master_data_tr, data_com_te
from Seq2Seq_Attn.reversed_input.data_com import MAX_LENGTH, pairs_tr as pairs1, pairs_te as pairs2, master_data_te

use_cuda = torch.cuda.is_available() #

test_accs = np.zeros((5,2))

for i in range(1):
    print("*****Starting run {}*****".format(i))
    hidden_size = 256
    encoder1 = EncoderRNN(input_lang1.n_words, hidden_size, use_cuda)
    attn_decoder1 = BahdanauAttnDecoderRNN("concat", hidden_size, output_lang1.n_words, use_cuda,
                                   1, dropout_p=0.1)

    if use_cuda:
        encoder1 = encoder1.cuda()
        attn_decoder1 = attn_decoder1.cuda()

    test_acc = trainIters(encoder1, attn_decoder1, 500, training_pairs,test_pairs, use_cuda, print_every=10, plot_every=10)
    test_accs[i,0] = i
    test_accs[i,1] = test_acc
# df = pd.DataFrame(test_accs,columns=['Experiment No.', 'Final Test Accuracy'])
# df.to_csv("final_test_acc_wo_attn.csv")

print("Evalualte on Training Data")
evaluateRandomly(encoder1, attn_decoder1, pairs1,input_lang1,output_lang1, use_cuda)


# for i in range(master_data_tr.shape[0]):
#     ipt_sentence = master_data_tr[i,0]
#     name = 'train'+'{}'.format(i)
#     evaluateAndShowAttention(ipt_sentence, encoder1, attn_decoder1, master_data_tr, input_lang1, output_lang1, use_cuda,name)
# print('')

print("Evaluate on Test Data")
evaluateRandomly(encoder1, attn_decoder1, pairs2,input_lang2,output_lang2, use_cuda)

for i in range(master_data_te.shape[0]):
    ipt_sentence = master_data_te[i,0]
    if(len(ipt_sentence.split(' '))==2):
        continue
    else:
        name = 'test'+'{}'.format(i)
        evaluateAndShowAttention(ipt_sentence, encoder1, attn_decoder1, master_data_te, input_lang2, output_lang2, use_cuda,name)

# evaluateAndShowAttention("00 t2",encoder1,attn_decoder1,master_data_te, input_lang2, output_lang2, use_cuda, "test1")
# evaluateAndShowAttention("01 t3",encoder1,attn_decoder1,master_data_te, input_lang2, output_lang2, use_cuda, "test1")
#
# evaluateAndShowAttention("11 t4 t3",encoder1,attn_decoder1,master_data_te, input_lang2, output_lang2, use_cuda, "test1")
# evaluateAndShowAttention("11 t2 t3",encoder1,attn_decoder1,master_data_te, input_lang2, output_lang2, use_cuda, "test1")
# evaluateAndShowAttention("01 t4 t2",encoder1,attn_decoder1,master_data_te, input_lang2, output_lang2, use_cuda, "test1")
# evaluateAndShowAttention("01 t3 t1",encoder1,attn_decoder1,master_data_te, input_lang2, output_lang2, use_cuda, "test1")
#
# evaluateAndShowAttention("11 t1 t1",encoder1,attn_decoder1,master_data_te, input_lang2, output_lang2, use_cuda, "test1")
# evaluateAndShowAttention("00 t2 t2",encoder1,attn_decoder1,master_data_te, input_lang2, output_lang2, use_cuda, "test1")
