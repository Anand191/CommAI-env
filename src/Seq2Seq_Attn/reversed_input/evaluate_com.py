import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
from torch.autograd import Variable
from Seq2Seq_Attn.reversed_input.data_com import variableFromSentence
from Seq2Seq_Attn.reversed_input.data_com import SOS_token, EOS_token, MAX_LENGTH
import numpy as np
import random

def evaluate(encoder, decoder, sentence, input_lang, output_lang, use_cuda=False, max_length=MAX_LENGTH):
    input_variable = variableFromSentence(input_lang, sentence)
    input_length = input_variable.size()[0]

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

    return decoded_words, decoder_attentions[:di+1,:len(encoder_outputs)]

def evaluateRandomly(encoder, decoder, pairs, input_lang, output_lang,use_cuda, n=5):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0], input_lang, output_lang, use_cuda)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


def showAttention(input_sentence, output_words, attentions,name):
    # Set up figure with colorbar
    attentions = attentions.numpy()
    #attentions = np.exp(attentions)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions, cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +['<EOS>'], rotation=90) #
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.savefig("{}.png".format(name))
    #plt.show()


def evaluateAndShowAttention(input_sentence,encoder1,attn_decoder1, master_data, input_lang, output_lang, use_cuda,name):

    output_words, attentions = evaluate(encoder1, attn_decoder1, input_sentence, input_lang, output_lang, use_cuda)
    #print(np.exp(attentions.numpy()))
    ipt = input_sentence.split(' ')
    nis = ''
    if(len(ipt)==2):
        row = np.where(master_data[:,0]==input_sentence)[0]
        tgt = master_data[row,1][0].split(' ')[0]
        nis = ipt[0]+' '+ipt[1] + '({})'.format(tgt)
    else:
        ipt2 = ipt[:-1]
        temp = ipt2[0]+' '+ipt2[1]
        row = np.where(master_data[:, 0] == temp)[0]
        tgt = master_data[row, 1][0].split(' ')[0]
        ni = ipt2[0]  + ' ' + ipt2[1] + '({})'.format(tgt)
        temp2 = tgt + ' ' + ipt[-1]
        row2 = np.where(master_data[:, 0] == temp2)[0]
        tgt2 = master_data[row2, 1][0].split(' ')[0]
        nis = ni + ' ' + ipt[-1]+ '({})'.format(tgt2)
    # else:
    #     ipt0 = ipt[1:]
    #     ipt2 = ipt0[1:]
    #     temp = ipt2[0] + ' ' + ipt2[1]
    #     row = np.where(master_data[:, 0] == temp)[0]
    #     tgt = master_data[row, 1][0].split(' ')[0]
    #     ni = ipt2[0] + '({})'.format(tgt) + ' ' + ipt2[1]
    #     temp2 = ipt0[0] + ' ' + tgt
    #     row2 = np.where(master_data[:, 0] == temp2)[0]
    #     tgt2 = master_data[row2, 1][0].split(' ')[0]
    #     ni2 = ipt0[0] + '({})'.format(tgt2) + ' ' + ni
    #     temp3 = ipt[0] + ' '+tgt2
    #     row3 = np.where(master_data[:,0]==temp3)[0]
    #     tgt3 = master_data[row3,1][0].split(' ')[0]
    #     nis = ipt[0] + '({})'.format(tgt3) + ' ' +ni2
    row_t = np.where(master_data[:,0]==input_sentence)[0]
    target = master_data[row_t,1]
    print('input =', input_sentence)
    print('target =', target)
    print('output =', ' '.join(output_words))
    showAttention(nis, output_words, attentions,name)