import torch
from torch.autograd import Variable
from Seq2Seq_Attn.composed_atomic.data_com import SOS_token,MAX_LENGTH, EOS_token
import numpy as np


def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion1,
          criterion2,use_cuda= False, max_length=MAX_LENGTH):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    encoder_hidden = encoder.initHidden()
    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)
    # encoder_outputs = Variable(torch.zeros(input_length, encoder.hidden_size))
    # encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0
    l1 = 0
    l2 = 0
    acc = 0
    ponder_step = input_length - 1

    attn_targets = torch.FloatTensor(np.eye(max_length))

    # for ei in range(input_length):
    #     encoder_output, encoder_hidden = encoder(
    #         input_variable[ei], encoder_hidden)
    #     encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    # Without teacher forcing: use its own predictions as the next input
    for di in range(target_length-2):
        i = input_length-2
        loss1 = 0
        for p_step in range(ponder_step):
            attn_target = Variable(torch.nonzero(attn_targets[i])[0])
            attn_target = attn_target.cuda() if use_cuda else attn_target
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss1 += criterion1(decoder_attention.squeeze(0), attn_target)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input
            if (p_step == 0):
                loss += criterion2(decoder_output, target_variable[-2])
            if ni == EOS_token:
                break
            i -= 1
        loss += criterion2(decoder_output, target_variable[di])
        l1 = loss.data[0]
        l2 = loss1.data[0]
        loss += loss1

        tv,ti = decoder_output.data.topk(1)
        do = ti[0][0]
        chk = Variable(torch.LongTensor([do]))
        chk = chk.cuda() if use_cuda else chk
        if(chk.data[0] == target_variable[di].data[0]):
            acc = 1
        # print(loss)

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return (loss.data[0]/ponder_step,l1/ponder_step,l2/ponder_step,acc)
