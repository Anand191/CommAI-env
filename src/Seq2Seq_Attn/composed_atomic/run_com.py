from Seq2Seq_Attn.Model2 import EncoderRNN, BahdanauAttnDecoderRNN
import torch
from Seq2Seq_Attn.composed_atomic.composed_training  import trainIters
from Seq2Seq_Attn.composed_atomic.evaluate_com import evaluateRandomly,evaluateAndShowAttention
from Seq2Seq_Attn.composed_atomic.data_com import input_lang_tr as input_lang1,output_lang_tr as output_lang1
from Seq2Seq_Attn.composed_atomic.data_com import input_lang_te as input_lang2,output_lang_te as output_lang2
from Seq2Seq_Attn.composed_atomic.data_com import training_pairs, test_pairs, master_data_tr, data_com_te
from Seq2Seq_Attn.composed_atomic.data_com import MAX_LENGTH, pairs_tr as pairs1, pairs_te as pairs2, master_data_te

use_cuda = torch.cuda.is_available() #


hidden_size = 64
encoder1 = EncoderRNN(input_lang1.n_words, hidden_size, use_cuda)
attn_decoder1 = BahdanauAttnDecoderRNN("concat", hidden_size, output_lang1.n_words, use_cuda,
                               1, dropout_p=0.1)

if use_cuda:
    encoder1 = encoder1.cuda()
    attn_decoder1 = attn_decoder1.cuda()

trainIters(encoder1, attn_decoder1, 100, training_pairs,test_pairs, use_cuda, print_every=10, plot_every=10)

print("Evalualte on Training Data")
evaluateRandomly(encoder1, attn_decoder1, pairs1,input_lang1,output_lang1, use_cuda)

# evaluateAndShowAttention("t1 t4 10",encoder1,attn_decoder1, master_data_tr, input_lang1,output_lang1, use_cuda)
# evaluateAndShowAttention("t3 t4 01",encoder1,attn_decoder1, master_data_tr, input_lang1,output_lang1, use_cuda)
# evaluateAndShowAttention("t4 t2 00",encoder1,attn_decoder1, master_data_tr, input_lang1,output_lang1, use_cuda)
# evaluateAndShowAttention("t1 t2 11",encoder1,attn_decoder1, master_data_tr, input_lang1,output_lang1, use_cuda)
# evaluateAndShowAttention("t1 11",encoder1,attn_decoder1, master_data_tr, input_lang1,output_lang1, use_cuda)
# evaluateAndShowAttention("t4 00",encoder1,attn_decoder1, master_data_tr, input_lang1,output_lang1, use_cuda)
# evaluateAndShowAttention("t3 t2 10",encoder1,attn_decoder1, master_data_tr, input_lang1,output_lang1, use_cuda)
for i in range(master_data_tr.shape[0]):
    ipt_sentence = master_data_tr[i,0]
    name = 'train'+'{}'.format(i)
    evaluateAndShowAttention(ipt_sentence, encoder1, attn_decoder1, master_data_tr, input_lang1, output_lang1, use_cuda,name)
print('')

print("Evaluate on Test Data")
evaluateRandomly(encoder1, attn_decoder1, pairs2,input_lang2,output_lang2, use_cuda)

for i in range(master_data_te.shape[0]):
    ipt_sentence = master_data_te[i,0]
    name = 'test'+'{}'.format(i)
    evaluateAndShowAttention(ipt_sentence, encoder1, attn_decoder1, master_data_tr, input_lang2, output_lang2, use_cuda,name)

# evaluateAndShowAttention("t1 t4 11",encoder1,attn_decoder1,master_data_te, input_lang1, output_lang1, use_cuda)
# evaluateAndShowAttention("t3 t4 00",encoder1,attn_decoder1,master_data_te, input_lang1, output_lang1, use_cuda)
# evaluateAndShowAttention("t4 t2 10",encoder1,attn_decoder1,master_data_te, input_lang1, output_lang1, use_cuda)
# evaluateAndShowAttention("t1 t2 01",encoder1,attn_decoder1,master_data_te, input_lang1, output_lang1, use_cuda)
# evaluateAndShowAttention("t3 t2 11",encoder1,attn_decoder1,master_data_te, input_lang1, output_lang1, use_cuda)
# evaluateAndShowAttention("t1 11",encoder1,attn_decoder1, master_data_te, input_lang1,output_lang1, use_cuda)
# evaluateAndShowAttention("t4 00",encoder1,attn_decoder1, master_data_te, input_lang1,output_lang1, use_cuda)