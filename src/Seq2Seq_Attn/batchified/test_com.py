import torch
from torch.autograd import Variable
from Seq2Seq_Attn.batchified.data_com_new import MAX_LENGTH, SOS_token
from Seq2Seq_Attn.batchified.metrics import Metrics
import numpy as np

def test(encoder, decoder, input_variable, target_variable,criterion2, use_cuda = False,max_length=MAX_LENGTH,
         use_copy = True, use_attn = True, use_interim = False, train_attn = True):

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
    interim_loss = torch.tensor(0.0).cuda() if use_cuda else torch.tensor(0.0)
    target_loss = 0
    losses = {'final_target_loss':0,'copy_loss': 0, 'attn_loss': 0, 'interim_loss': 0}
    accuracies = {'word_level': 0, 'seq_level': 0, 'final_target': 0}
    final_outputs = []
    ########################################################################################################################
    for di in range(ponder_step):
        target_weight = Variable(attn_targets[i]).unsqueeze(0).unsqueeze(0)
        target_weight = target_weight.cuda() if use_cuda else target_weight
        attn_target = Variable(torch.nonzero(attn_targets[i])[0])
        attn_target = attn_target.cuda() if use_cuda else attn_target
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden,
                                                                    encoder_outputs, target_weight, train_attn)
        attn_loss += criterion2(torch.log(decoder_attention).squeeze(0), attn_target)
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if(di ==0):
            copy_loss += criterion2(decoder_output, target_variable[0])
        elif (di < ponder_step-1):
            interim_loss += criterion2(decoder_output, target_variable[di])
        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input
        final_outputs.append(decoder_input.data[0][0])
        i += 1
    target_loss += criterion2(decoder_output, target_variable[-1])

    losses['final_target_loss'] = target_loss.item() #data[0] #/ ponder_step
    if (use_copy):
        target_loss += copy_loss
        losses['copy_loss'] = copy_loss.item() #data[0] #/ ponder_step

    if (use_attn):
        target_loss += (attn_loss/ponder_step)
        losses['attn_loss'] = attn_loss.item()/ponder_step #data[0] / ponder_step

    if (use_interim):
        target_loss += (interim_loss/(ponder_step-2))
        losses['interim_loss'] = interim_loss.item()/(ponder_step-2) #data[0] / (ponder_step-1)
    metrics = Metrics()
    target_outputs = target_variable.cpu().data.squeeze(-1).numpy().tolist()
    #target_outputs = target_variable.cpu().data[:-1].squeeze(-1).numpy().tolist()
    accuracies['word_level'] = metrics.word_level(final_outputs, target_outputs)
    accuracies['seq_level'] = metrics.seq_level(final_outputs, target_outputs)
    accuracies['final_target'] = metrics.final_target(final_outputs, target_outputs)

    return (target_loss.item(),losses,accuracies) #/ponder_step data[0]