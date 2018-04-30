import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import numpy as np
from Seq2Seq_Attn.batchified.data_com_new import MAX_LENGTH, SOS_token
from Seq2Seq_Attn.batchified.metrics import Metrics


def inference(encoder, decoder, input_variable, target_variable,criterion2, use_cuda = False,max_length=MAX_LENGTH,
              use_copy=True, use_attn=True, use_interim=False, train_attn=True):

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]


    encoder.eval()
    decoder.eval()

    encoder_hidden = encoder.initHidden()
    encoder_outputs, encoder_hidden = encoder(input_variable.transpose(0,1), encoder_hidden)

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    attn_targets = torch.FloatTensor(np.eye(input_length))
    ########################################################################################################################
    i = 0
    ponder_step = input_length
    attn_loss = 0
    copy_loss = 0
    interim_loss = 0
    target_loss = 0
    losses = {'final_target_loss': 0, 'copy_loss': 0, 'attn_loss': 0, 'interim_loss': 0}
    accuracies = {'word_level': 0, 'seq_level': 0, 'final_target': 0}
    final_outputs = []
    ########################################################################################################################
    for di in range(ponder_step):
        target_weight = Variable(attn_targets[i]).unsqueeze(0).unsqueeze(0)
        target_weight = target_weight.cuda() if use_cuda else target_weight
        attn_target = Variable(torch.nonzero(attn_targets[i])[0])
        attn_target = attn_target.cuda() if use_cuda else attn_target
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs,
                                                                    target_weight, train_attn)
        attn_loss += criterion2(torch.log(decoder_attention).squeeze(0), attn_target)
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if(di ==0):
            copy_loss += criterion2(decoder_output, target_variable[0])
        else:
            interim_loss += criterion2(decoder_output, target_variable[di])
        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input
        final_outputs.append(decoder_input.data[0][0])
        i += 1

    target_loss += criterion2(decoder_output, target_variable[-1])
    losses['final_target_loss'] = target_loss.item() #data[0] #/ ponder_step
    # tv, ti = decoder_output.data.topk(1)
    # do = ti[0][0]
    # chk = Variable(torch.LongTensor([do]))
    # chk = chk.cuda() if use_cuda else chk
    # if (chk.data[0] == target_variable[1].data[0]):
    #     acc = 1
    # else:
    #     acc = 0
    if (use_copy):
        target_loss += copy_loss
        losses['copy_loss'] = copy_loss.item() #.data[0] #/ ponder_step

    if (use_attn):
        target_loss += (attn_loss/ponder_step)
        losses['attn_loss'] = attn_loss.item() #data[0] / ponder_step

    if (use_interim):
        target_loss += (interim_loss/ (ponder_step-1))
        losses['interim_loss'] = interim_loss.item() #data[0] / (ponder_step-1)
    metrics = Metrics()
    target_outputs = target_variable.cpu().data.squeeze(-1).numpy().tolist()
    #target_outputs = target_variable.cpu().data[:-1].squeeze(-1).numpy().tolist()
    accuracies['word_level'] = metrics.word_level(final_outputs, target_outputs)
    accuracies['seq_level'] = metrics.seq_level(final_outputs, target_outputs)
    accuracies['final_target'] = metrics.final_target(final_outputs, target_outputs)

    return (target_loss.item(), losses, accuracies) #/ ponder_step data[0]



def inferIters(encoder, decoder, infer_pairs, use_cuda=False, use_copy=True, use_attn=True, use_interim=False,
               train_attn = True,name='Test'):
    wc_infer = 0
    for tr in infer_pairs:
        wc_infer += (tr[1].size()[0]) # -1)
    print_loss_total = 0  # Reset every print_every
    print_acc_total = 0
    target_inf, copy_inf, interim_inf, attn_inf = 0, 0, 0, 0
    word_inf, seq_inf = 0, 0
    criterion2 = nn.NLLLoss()
    test_l, test_a = 0, 0
    tl0, tl1, tl2, tl3 = 0, 0, 0, 0
    ta1, ta2 = 0, 0
    for j in range(len(infer_pairs)):
        infer_pair = infer_pairs[j]
        input_var = infer_pair[0]
        target_var = infer_pair[1]

        loss_t, other_t, acc_t = inference(encoder, decoder, input_var, target_var, criterion2, use_cuda,
                                      use_copy=use_copy, use_attn=use_attn, use_interim=use_interim,
                                      train_attn=train_attn)
        test_l += loss_t
        tl0 += other_t['final_target_loss']
        tl1 += other_t['copy_loss']
        tl2 += other_t['attn_loss']
        tl3 += other_t['interim_loss']

        test_a += acc_t['final_target']
        ta1 += acc_t['word_level']
        ta2 += acc_t['seq_level']

    print_acc_total += (test_a/len(infer_pairs))
    word_inf += (ta1 / wc_infer)
    seq_inf += (ta2 / len(infer_pairs))

    print_loss_total += (test_l/len(infer_pairs))
    target_inf += (tl0 / len(infer_pairs))
    copy_inf += (tl1 / len(infer_pairs))
    attn_inf += (tl2 / len(infer_pairs))
    interim_inf += (tl3 / len(infer_pairs))

    name = name.split('.')[0]
    name = name.split('_')[-1]
    print('')
    # print('%s  %s: %.4f %s: %.4f %s: %.4f %s:%.4f'
    #       % (name,
    #          "Average Final Target Loss", target_inf,
    #          "Average Copy Loss", copy_inf,
    #          "Attention Loss", attn_inf,
    #          "Average Intermediate Loss", interim_inf
    #          ))
    # print('')
    print('%s %s:%.4f %s:%.4f %s:%.4f' % (name,
                                          "Word Level Accuracy", word_inf,
                                          "Sequence Level Accuracy", seq_inf,
                                          "Final Target Accuracy", print_acc_total
                                          ))
    print('************************************************************************************************************')
    name = name[0:-1]

    return(name, word_inf,seq_inf,print_acc_total)