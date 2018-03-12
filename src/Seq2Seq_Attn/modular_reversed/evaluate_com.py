import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl
import torch
from torch.autograd import Variable
from Seq2Seq_Attn.modular_reversed.data_com_new import SOS_token, MAX_LENGTH, EOS_token
import numpy as np
import random

def evaluate(encoder, decoder, sentence, input_lang, output_lang, variableFromSentence, use_cuda=False, max_length=MAX_LENGTH):
    input_variable = variableFromSentence(input_lang, sentence)
    input_length = input_variable.size()[0]

    decoder.eval()

    encoder_hidden = encoder.initHidden()
    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)
    # encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    # encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    # for ei in range(R):
    #     encoder_output, encoder_hidden = encoder(input_variable[ei],
    #                                              encoder_hidden)
    #     encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(input_length-1):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        #print(decoder_attention.squeeze(0))
        decoder_attentions[di,:decoder_attention.size(2)] = decoder_attention.squeeze(0).squeeze(0).cpu().data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_words, decoder_attentions[:di+1,:len(encoder_outputs)-1]

def evaluateRandomly(encoder, decoder, pairs, input_lang, output_lang,use_cuda, n=5):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0], input_lang, output_lang, use_cuda)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


def showAttention(input_sentence, output_words, attentions,name,colour):
    # Set up figure with colorbar
    attentions = attentions.numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.yaxis.tick_right()
    cax = ax.matshow(attentions, cmap='bone', vmin=0, vmax=1)
    #fig.colorbar(cax)
    cbaxes = fig.add_axes([0.05, 0.1, 0.03, 0.8])
    cb = plt.colorbar(cax, cax=cbaxes)
    cbaxes.yaxis.set_ticks_position('left')

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') , rotation=0) #+['<EOS>']
    ax.set_yticklabels([''] + output_words)

    #Colour ticks
    for ytick, color in zip(ax.get_yticklabels()[1:], colour):
        ytick.set_color(color)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    #X and Y labels
    ax.set_xlabel("INPUT")
    ax.set_ylabel("OUTPUT")
    ax.yaxis.set_label_position('right')
    ax.xaxis.set_label_position('top')

    plt.savefig("{}.png".format(name))
    plt.close(fig)
    #plt.show()


def evaluateAndShowAttention(input_sentence,encoder1,attn_decoder1, master_data, input_lang, output_lang, use_cuda,vfs,name):

    output_words, attentions = evaluate(encoder1, attn_decoder1, input_sentence, input_lang, output_lang,vfs, use_cuda)

    ipt = input_sentence.split(' ')
    nis = ipt[0]
    tgt = ipt[0]
    i = 1
    colour = []
    while(i<=len(ipt)):
        if(output_words[i-1]==tgt):
            colour.append('g')
        else:
            colour.append('r')
        if(i==len(ipt)):
            break
        temp = tgt + ' ' + ipt[i]
        row = np.where(master_data[:, 0] == temp)[0]
        tgt = master_data[row, 1][0].split(' ')[1]
        nis += ' ' +ipt[i] + '({})'.format(tgt)
        i+=1
    row_t = np.where(master_data[:,0]==input_sentence)[0]
    target = master_data[row_t,1]
    # print('input =', input_sentence)
    # print('target =', target)
    # print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions,name,colour)