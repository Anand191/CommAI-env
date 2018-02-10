import torch.nn as nn
import matplotlib.ticker as ticker
from torch import optim
import matplotlib.pyplot as plt
from Seq2Seq_Attn.train import train

import time
import math


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


def trainIters(encoder, decoder, n_iters, training_pairs, print_every=1, plot_every=5, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    avg_l1 = 0
    avg_l2 = 0

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        temp_loss = 0
        tl1,tl2 = 0, 0
        for j in range(len(training_pairs)):
            training_pair = training_pairs[j]
            input_variable = training_pair[0]
            target_variable = training_pair[1]

            loss,l1,l2 = train(input_variable, target_variable, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion1, criterion2)
            temp_loss += loss
            tl1+=l1
            tl2 += l2
        print_loss_total += (temp_loss/len(training_pairs))
        plot_loss_total += (temp_loss/len(training_pairs))

        avg_l1 += (tl1/len(training_pairs))
        avg_l2 += (tl2 / len(training_pairs))

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            avg_l1_avg = avg_l1/ print_every
            avg_l2_avg = avg_l2/ print_every
            print_loss_total = 0
            avg_l1,avg_l2 = 0,0
            print('%s (%d %d%%) %.4f %.4f %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg,avg_l1_avg,avg_l2_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.show()

