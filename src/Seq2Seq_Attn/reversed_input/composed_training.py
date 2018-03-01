import torch.nn as nn
import matplotlib.ticker as ticker
from torch import optim
import matplotlib.pyplot as plt
from Seq2Seq_Attn.reversed_input.train_com import train
from Seq2Seq_Attn.reversed_input.test_com import test
from Seq2Seq_Attn.reversed_input.checkpoint import checkpoint

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


def trainIters(encoder, decoder, n_iters, training_pairs, test_pairs, use_cuda=False, print_every=1, plot_every=5, learning_rate=0.01):
    start = time.time()
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

    avg_l1 = 0
    avg_l2 = 0

    best_acc = 0
    echk = './Encoder_Weights'
    dchk = './Decoder_Weights'

    saver_e = checkpoint(encoder,echk)
    saver_d = checkpoint(decoder,dchk)

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        temp_loss = 0
        tl1,tl2 = 0, 0
        temp_acc = 0
        test_l, test_a = 0, 0
        shuffle(training_pairs)
        for j in range(len(training_pairs)):
            training_pair = training_pairs[j]
            input_variable = training_pair[0]
            target_variable = training_pair[1]

            loss,l1,l2,acc = train(input_variable, target_variable, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion1, criterion2, use_cuda)
            temp_loss += loss
            tl1+=l1
            tl2 += l2
            temp_acc += acc
        print_loss_total += (temp_loss/len(training_pairs))
        plot_loss_total += (temp_loss/len(training_pairs))

        print_acc_total += (temp_acc/len(training_pairs))
        plot_acc_total += (temp_acc / len(training_pairs))

        avg_l1 += (tl1/len(training_pairs))
        avg_l2 += (tl2 / len(training_pairs))

        for j in range(len(test_pairs)):
            test_pair = test_pairs[j]
            input_var = test_pair[0]
            target_var = test_pair[1]

            loss_t, acc_t, _ = test(encoder,decoder,input_var,target_var,criterion2, use_cuda)
            test_l += loss_t
            test_a += acc_t
        print_test_acc += (test_a/len(test_pairs))
        plot_test_acc += (test_a/len(test_pairs))

        print_test_loss += (test_l/len(test_pairs))
        plot_test_loss += (test_l / len(test_pairs))


        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            avg_l1_avg = avg_l1/ print_every
            avg_l2_avg = avg_l2/ print_every
            print_acc_avg = print_acc_total / print_every

            print_test_l_avg = print_test_loss / print_every
            print_test_a_avg = print_test_acc / print_every

            print_loss_total = 0
            print_acc_total = 0
            print_test_acc = 0
            print_test_loss = 0
            avg_l1,avg_l2 = 0,0

            print('%s %s (%d %d%%) %.4f %.4f %.4f %.4f' % ("Train",timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg,avg_l1_avg,avg_l2_avg, print_acc_avg))
            print('')
            print('%s %s (%d %d%%) %.4f %.4f' % ("Test", timeSince(start, iter / n_iters),
                                                           iter, iter / n_iters * 100, print_test_l_avg,print_test_a_avg ))
            print('')

            saver_e.save(print_test_a_avg,best_acc,iter+1)
            saver_d.save(print_test_a_avg, best_acc, iter + 1)

            if(print_test_a_avg > best_acc):
                best_acc = print_test_a_avg

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

    showPlot(plot_losses)
    showAcc(plot_accuracy)
    showPlot(plot_tloss)
    showAcc(plot_tacc)
    return (print_test_a_avg)


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.show()

def showAcc(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.show()

