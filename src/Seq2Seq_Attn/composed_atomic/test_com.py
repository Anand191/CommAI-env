import torch
from torch.autograd import Variable
from Seq2Seq_Attn.composed_atomic.data_com import MAX_LENGTH, SOS_token, EOS_token, output_lang_te as output_lang

def test(encoder, decoder, input_variable, target_variable,criterion2, use_cuda = False,max_length=MAX_LENGTH):

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    encoder_hidden = encoder.initHidden()
    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)

    # encoder_outputs = Variable(torch.zeros(input_length, encoder.hidden_size))
    # encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0
    acc = 0

    # for ei in range(input_length):
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
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]

        if(di ==0):
            loss += criterion2(decoder_output, target_variable[-2])
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    loss += criterion2(decoder_output, target_variable[0])
    tv, ti = decoder_output.data.topk(1)
    do = ti[0][0]
    chk = Variable(torch.LongTensor([do]))
    chk = chk.cuda() if use_cuda else chk
    if (chk.data[0] == target_variable[0].data[0]):
        acc = 1
    return (loss.data[0]/(target_length-1),acc,decoder_attentions[:di + 1])