import torch
from torch.autograd import Variable
import numpy as np
from Seq2Seq_Attn.batchified.metrics import Metrics
from Seq2Seq_Attn.batchified.data_com_new import SOS_token,MAX_LENGTH


def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion2,
          use_cuda= False, max_length=MAX_LENGTH, use_copy=True, use_attn=True, use_interim=False,
          train_attn=True, clip=0.25, lr=0.001):


    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder.train()
    decoder.train()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    encoder_hidden = encoder.initHidden()
    encoder_outputs, encoder_hidden = encoder(input_variable.transpose(0,1), encoder_hidden)

    attn_targets = torch.FloatTensor(np.eye(input_length))

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))

    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    # Without teacher forcing: use its own predictions as the next input
    #for di in range(target_length-1):
########################################################################################################################
    i = 0
    ponder_step = input_length - 1
    attn_loss = 0
    copy_loss = 0
    interim_loss = 0
    target_loss = 0
    losses = {'final_target_loss':0,'copy_loss':0, 'attn_loss':0, 'interim_loss':0}
    accuracies = {'word_level':0, 'seq_level':0, 'final_target':0}
    final_outputs = []
########################################################################################################################

    for p_step in range(ponder_step):
        target_weight = Variable(attn_targets[i]).unsqueeze(0).unsqueeze(0)
        target_weight = target_weight.cuda() if use_cuda else target_weight
        attn_target = Variable(torch.nonzero(attn_targets[i])[0])
        attn_target = attn_target.cuda() if use_cuda else attn_target
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden,
                                                                    encoder_outputs, target_weight, train_attn)
        attn_loss += criterion2(torch.log(decoder_attention).squeeze(0), attn_target)
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input
        final_outputs.append(decoder_input.data[0][0])
        if (p_step == 0):
            copy_loss += criterion2(decoder_output, target_variable[0])
        else:
            interim_loss += criterion2(decoder_output, target_variable[p_step])
        i += 1
    target_loss += criterion2(decoder_output, target_variable[-2])

    losses['final_target_loss'] = target_loss.data[0]/ponder_step
    if(use_copy):
        target_loss += copy_loss
        losses['copy_loss'] = copy_loss.data[0]/ponder_step

    if(use_attn):
        target_loss += attn_loss
        losses['attn_loss'] = attn_loss.data[0]/ponder_step

    if(use_interim):
        target_loss += interim_loss
        losses['interim_loss'] = interim_loss.data[0]/ponder_step

    # tv,ti = decoder_output.data.topk(1)
    # do = ti[0][0]
    # chk = Variable(torch.LongTensor([do]))
    # chk = chk.cuda() if use_cuda else chk
    # if(chk.data[0] == target_variable[1].data[0]):
    #     acc = 1
    # else:
    #     acc = 0
    metrics = Metrics()
    target_outputs = target_variable.cpu().data[:-1].squeeze(-1).numpy().tolist()
    accuracies['seq_level'] = metrics.seq_level(final_outputs, target_outputs)
    accuracies['word_level'] = metrics.word_level(final_outputs, target_outputs)
    accuracies['final_target'] = metrics.final_target(final_outputs, target_outputs)

    target_loss.backward()
########################################################################################################################
    #gradient clipping
    # torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
    # for p in encoder.parameters():
    #     p.data.add_(-lr, p.grad.data)
    #
    # torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)
    # for p in decoder.parameters():
    #     p.data.add_(-lr, p.grad.data)
########################################################################################################################
    encoder_optimizer.step()
    decoder_optimizer.step()

    return (target_loss.data[0]/ponder_step,losses,accuracies)
