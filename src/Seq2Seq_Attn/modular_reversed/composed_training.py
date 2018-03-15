import torch.nn as nn
import matplotlib.ticker as ticker
from torch import optim
import matplotlib.pyplot as plt
from Seq2Seq_Attn.modular_reversed.train_com import train
from Seq2Seq_Attn.modular_reversed.test_com import test
from Seq2Seq_Attn.modular_reversed.checkpoint import checkpoint

import time
import math
from random import shuffle


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def trainIters(encoder, decoder, n_iters, training_pairs, test_pairs, use_cuda=False, print_every=1, plot_every=5,
               learning_rate=0.01, use_copy=True, use_attn=True, use_interim=False, clip=0.25):
    start = time.time()

    wc_train = 0
    wc_dev = 0
    for tr in training_pairs:
        wc_train += (tr[1].size()[0] -1)
    for te in test_pairs:
        wc_dev += (te[1].size()[0] -1)

    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    plot_accuracy = []
    print_acc_total = 0
    plot_acc_total = 0

    plot_tloss = []
    print_test_loss = 0
    plot_test_loss = 0

    plot_tacc = []
    print_test_acc = 0
    plot_test_acc = 0

    target_train, copy_train,interim_train,attn_train = 0,0,0,0
    target_test, copy_test, interim_test, attn_test = 0,0,0,0

    word_train, seq_train = 0,0
    word_test, seq_test = 0,0

    ####################################################################################################################
    #not being used right now
    # best_acc = 0
    # echk = './Encoder_Weights'
    # dchk = './Decoder_Weights'
    #
    # saver_e = checkpoint(encoder,echk)
    # saver_d = checkpoint(decoder,dchk)
    ####################################################################################################################
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    criterion2 = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        temp_loss = 0
        l0,l1,l2,l3 = 0, 0, 0, 0
        temp_acc = 0
        a1, a2 = 0, 0
        shuffle(training_pairs)
        for j in range(len(training_pairs)):
            training_pair = training_pairs[j]
            input_variable = training_pair[0]
            target_variable = training_pair[1]

            loss,other,acc = train(input_variable, target_variable, encoder,decoder, encoder_optimizer,
                                   decoder_optimizer, criterion2, use_cuda, use_copy = use_copy, use_attn = use_attn,
                                   use_interim = use_interim, clip=clip, lr=learning_rate
                                   )
            temp_loss += loss
            l0 += other['final_target_loss']
            l1 += other['copy_loss']
            l2 += other['attn_loss']
            l3 += other['interim_loss']

            temp_acc += acc['final_target']
            a1 += acc['word_level']
            a2 += acc['seq_level']

        print_loss_total += (temp_loss/len(training_pairs))
        plot_loss_total += (temp_loss/len(training_pairs))

        print_acc_total += (temp_acc/len(training_pairs))
        plot_acc_total += (temp_acc / len(training_pairs))

        target_train += (l0/len(training_pairs))
        copy_train += (l1/len(training_pairs))
        attn_train += (l2 / len(training_pairs))
        interim_train += (l3 / len(training_pairs))

        word_train += (a1/wc_train)
        seq_train += (a2/len(training_pairs))

        # shuffle(test_pairs)
        test_l, test_a = 0, 0
        dl0, dl1, dl2, dl3 = 0, 0, 0, 0
        da1, da2 = 0, 0
        for j in range(len(test_pairs)):
            test_pair = test_pairs[j]
            input_var = test_pair[0]
            target_var = test_pair[1]

            loss_t, other_t, acc_t = test(encoder, decoder, input_var, target_var, criterion2, use_cuda,
                                          use_copy = use_copy, use_attn = use_attn, use_interim = use_interim
                                          )

            test_l += loss_t
            dl0 += other_t['final_target_loss']
            dl1 += other_t['copy_loss']
            dl2 += other_t['attn_loss']
            dl3 += other_t['interim_loss']

            test_a += acc_t['final_target']
            da1 += acc_t['word_level']
            da2 += acc_t['seq_level']

        print_test_acc += (test_a / len(test_pairs))
        plot_test_acc += (test_a / len(test_pairs))

        print_test_loss += (test_l / len(test_pairs))
        plot_test_loss += (test_l / len(test_pairs))

        target_test += (dl0/len(test_pairs))
        copy_test += (dl1/len(test_pairs))
        attn_test += (dl2/len(test_pairs))
        interim_test += (dl3/len(test_pairs))

        word_test += (da1/wc_dev)
        seq_test += (da2/len(test_pairs))

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            train_target_avg = target_train/ print_every
            train_copy_avg = copy_train/ print_every
            train_attn_avg = attn_train/ print_every
            train_interim_avg = interim_train/ print_every

            print_acc_avg = print_acc_total / print_every
            train_word_avg = word_train/ print_every
            train_seq_avg = seq_train/ print_every

            print_test_l_avg = print_test_loss / print_every
            dev_target_avg = target_test/ print_every
            dev_copy_avg = copy_test/ print_every
            dev_attn_avg = attn_test/ print_every
            dev_interim_avg = interim_test/ print_every

            print_test_a_avg = print_test_acc / print_every
            dev_word_avg = word_test/ print_every
            dev_seq_avg = seq_test/ print_every

            print_loss_total = 0
            target_train, copy_train, attn_train, interim_train = 0, 0, 0, 0
            print_acc_total = 0
            word_train, seq_train = 0, 0

            print_test_acc = 0
            target_test, copy_test, attn_test, interim_test = 0, 0, 0, 0
            print_test_loss = 0
            word_test, seq_test = 0, 0


            print('%s %s (%d %d%%) %s: %.4f %s: %.4f %s: %.4f %s: %.4f %s:%.4f'
                  % ("Train",timeSince(start, iter / n_iters),iter, iter / n_iters * 100,
                     "Average Total Loss", print_loss_avg,
                     "Average Final Target Loss", train_target_avg,
                     "Average Copy Loss",train_copy_avg,
                     "Attention Loss", train_attn_avg,
                     "Average Intermediate Loss",train_interim_avg,
                     ))
            print('')
            print('%s %s (%d %d%%) %s:%.4f %s:%.4f %s:%.4f'% ("Train",timeSince(start, iter / n_iters),iter,
                                                              iter / n_iters * 100,
                                                              "Word Level Accuracy", train_word_avg,
                                                              "Sequence Level Accuracy", train_seq_avg,
                                                              "Final Target Accuracy", print_acc_avg
                                                              ))
            print('')
            print('%s %s (%d %d%%) %s: %.4f %s: %.4f %s: %.4f %s: %.4f %s:%.4f'
                  % ("Validation", timeSince(start, iter / n_iters), iter, iter / n_iters * 100,
                     "Average Total Loss", print_test_l_avg,
                     "Average Final Target Loss",dev_target_avg,
                     "Average Copy Loss", dev_copy_avg,
                     "Attention Loss", dev_attn_avg,
                     "Average Intermediate Loss", dev_interim_avg
                     ))
            print('')
            print('%s %s (%d %d%%) %s:%.4f %s:%.4f %s:%.4f' % ("Validation", timeSince(start, iter / n_iters), iter,
                                                               iter / n_iters * 100,
                                                               "Word Level Accuracy", dev_word_avg,
                                                               "Sequence Level Accuracy", dev_seq_avg,
                                                               "Final Target Accuracy", print_test_a_avg
                                                               ))
            print('************************************************************************************************************')
            print('')
########################################################################################################################
            # saver_e.save(print_test_a_avg,best_acc,iter+1)
            # saver_d.save(print_test_a_avg, best_acc, iter + 1)
            #
            # if(print_test_a_avg > best_acc):
            #     best_acc = print_test_a_avg
########################################################################################################################
        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_acc_avg = plot_acc_total/ plot_every
            plot_losses.append(plot_loss_avg)
            plot_accuracy.append(plot_acc_avg)

            plot_test_l_avg = plot_test_loss / print_every
            plot_test_a_avg = plot_test_acc / print_every
            plot_tloss.append(plot_test_l_avg)
            plot_tacc.append(plot_test_a_avg)

            plot_loss_total = 0
            plot_acc_total = 0
            plot_test_loss = 0
            plot_test_acc = 0

    # showPlot(plot_losses,'train_loss')
    # showAcc(plot_accuracy,'train_acc')
    # showPlot(plot_tloss,'valid_loss')
    # showAcc(plot_tacc,'valid_acc')
    # return (print_test_a_avg)


def showPlot(points, name):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    #plt.savefig('{}.png'.format(name))
    plt.show()

def showAcc(points, name):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    #plt.savefig('{}.png'.format(name))
    plt.show()

