import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from Seq2Seq_Attn.reversed_input.data_com import MAX_LENGTH, SOS_token, output_lang_if as output_lang, EOS_token


def inference(encoder, decoder, input_variable, target_variable,criterion2, use_cuda = False,max_length=MAX_LENGTH):

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    encoder_hidden = encoder.initHidden()
    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)

    loss = 0
    acc = 0

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(input_length-1):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        decoder_attentions[di, :decoder_attention.size(2)] = decoder_attention.squeeze(0).squeeze(0).cpu().data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]

        if(di ==0):
            loss += criterion2(decoder_output, target_variable[0])
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    loss += criterion2(decoder_output, target_variable[-2])
    tv, ti = decoder_output.data.topk(1)
    do = ti[0][0]
    chk = Variable(torch.LongTensor([do]))
    chk = chk.cuda() if use_cuda else chk
    if (chk.data[0] == target_variable[-2].data[0]):
        acc = 1
    else:
        acc = 0
    return (loss.data[0]/(input_length-1),acc,decoder_attentions[:di + 1])



def inferIters(encoder, decoder, infer_pairs, use_cuda=False):

    print_loss_total = 0  # Reset every print_every
    print_acc_total = 0
    criterion2 = nn.NLLLoss()
    test_l, test_a = 0, 0

    for j in range(len(infer_pairs)):
        infer_pair = infer_pairs[j]
        input_var = infer_pair[0]
        target_var = infer_pair[1]

        loss_t, acc_t, _ = inference(encoder,decoder,input_var,target_var,criterion2, use_cuda)
        test_l += loss_t
        test_a += acc_t
    print_acc_total += (test_a/len(infer_pairs))

    print_loss_total += (test_l/len(infer_pairs))

    return(print_loss_total,print_acc_total)