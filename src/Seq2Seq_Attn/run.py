from Seq2Seq_Attn.Model import EncoderRNN, AttnDecoderRNN
import torch
from Seq2Seq_Attn.seq2seq_training import trainIters
from Seq2Seq_Attn.evaluate import evaluateRandomly,evaluateAndShowAttention
from Seq2Seq_Attn.data_prep import training_pairs,input_lang,output_lang,MAX_LENGTH

use_cuda = torch.cuda.is_available()


hidden_size = 64
encoder1 = EncoderRNN(input_lang.n_words, hidden_size)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words,
                               1, dropout_p=0.1,max_length=MAX_LENGTH)

if use_cuda:
    encoder1 = encoder1.cuda()
    attn_decoder1 = attn_decoder1.cuda()

trainIters(encoder1, attn_decoder1, 100, training_pairs, print_every=10, plot_every=10)

evaluateRandomly(encoder1, attn_decoder1)


evaluateAndShowAttention("t1 11",encoder1,attn_decoder1)
evaluateAndShowAttention("t2 01",encoder1,attn_decoder1)
evaluateAndShowAttention("t3 00",encoder1,attn_decoder1)
evaluateAndShowAttention("t4 10",encoder1,attn_decoder1)