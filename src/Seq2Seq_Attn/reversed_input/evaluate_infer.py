from Seq2Seq_Attn.Model2 import EncoderRNN, BahdanauAttnDecoderRNN
from Seq2Seq_Attn.reversed_input.checkpoint import checkpoint
import torch
import numpy as np
from Seq2Seq_Attn.reversed_input.evaluate_com import evaluateRandomly,evaluateAndShowAttention
from Seq2Seq_Attn.reversed_input.data_com import input_lang_if as input_lang,output_lang_if as output_lang
from Seq2Seq_Attn.reversed_input.data_com import MAX_LENGTH, pairs_if as pairs, master_data_if, infer_pairs
from Seq2Seq_Attn.reversed_input.infer_com import inferIters

use_cuda = torch.cuda.is_available()

# hidden_size = 256
# encoder1 = EncoderRNN(input_lang.n_words, hidden_size, use_cuda)
# attn_decoder1 = BahdanauAttnDecoderRNN("concat", hidden_size, output_lang.n_words, use_cuda,
#                                    1, dropout_p=0.1)

path1 = './Encoder_Weights'
path2 = './Decoder_Weights'

cp1 = checkpoint.load(path1,use_cuda)
cp2 = checkpoint.load(path2,use_cuda)

encoder1 = cp1.model
attn_decoder1 = cp2.model

if use_cuda:
    encoder1 = encoder1.cuda()
    attn_decoder1 = attn_decoder1.cuda()


print("Evaluate on Inference Data")
evaluateRandomly(encoder1, attn_decoder1, pairs,input_lang,output_lang, use_cuda)

for i in range(0,master_data_if.shape[0]):
    ipt_sentence = master_data_if[i,0]
    if(len(ipt_sentence.split(' '))==2):
        continue
    else:
        name = './Infer_Results/infer'+'{}'.format(i)
        evaluateAndShowAttention(ipt_sentence, encoder1, attn_decoder1, master_data_if, input_lang, output_lang,
                                 use_cuda,name)

infer_loss, infer_acc = inferIters(encoder1,attn_decoder1, infer_pairs,use_cuda)
print('')
print('%s %.4f %.4f' % ("Inference", infer_loss, infer_acc))
print('')