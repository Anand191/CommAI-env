import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
from torch.autograd import Variable
from Seq2Seq_Attn.composed_atomic.data_com import variableFromSentence
from Seq2Seq_Attn.composed_atomic.data_com import SOS_token, EOS_token, MAX_LENGTH
import numpy as np
import random

use_cuda = torch.cuda.is_available()

def evaluate(encoder, decoder, sentence, input_lang, output_lang, max_length=MAX_LENGTH):
    input_variable = variableFromSentence(input_lang, sentence)
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    R = 0
    if(input_length < MAX_LENGTH):
        R = input_length
    else:
        R = MAX_LENGTH

    for ei in range(R):
        encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                 encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    # if (input_length < MAX_LENGTH):
    #     max_length = input_length


    for di in range(R-1):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs, target_attn=0, loss_func='na', training=False)
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_words, decoder_attentions[:di + 1]

def evaluateRandomly(encoder, decoder, pairs, input_lang, output_lang, n=5):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0], input_lang, output_lang)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +['<EOS>'], rotation=90) #
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.show()


def evaluateAndShowAttention(input_sentence,encoder1,attn_decoder1, master_data, input_lang, output_lang):
    output_words, attentions = evaluate(encoder1, attn_decoder1, input_sentence, input_lang, output_lang)
    print(attentions)
    row_t = np.where(master_data[:,0]==input_sentence)[0]
    target = master_data[row_t,1]
    print('input =', input_sentence)
    print('target =', target)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)