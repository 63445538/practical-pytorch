from data import *
from model import *
from torch.autograd import Variable
import torch
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from masked_cross_entropy import *
import time
import datetime
import math
import argparse
import os

argparser = argparse.ArgumentParser()
argparser.add_argument('filename', type=str, default="data/english.txt")

args = argparser.parse_args()
file, ext = os.path.splitext(args.filename)

ENCODE_FILE = file + "_encoder1.pt"
DECODE_FILE = file + "_attn_decoder1.pt"


def indexes_from_sentence(lang, sentence):
    return lang.indexes(sentence)

def variable_from_sentence(lang, sentence):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1,1))
    if USE_CUDA:
        return result.cuda()
    else:
        return result

teacher_forcing_ratio = 0.5

def train(input_variable, 
          encoder, 
          decoder, 
          encoder_optimizer, 
          decoder_optimizer, 
          criterion, 
          max_length=MAX_LENGTH):

    target_variable = input_variable
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = min(input_variable.size()[0], max_length)
    target_length = input_length
    # max_length是一句话允许出现的最大长度，目前设为25.超过此长度后续将被截断丢弃
    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    #print("encoder_outputs shape {}".format(encoder_outputs.data.numpy().shape))
    encoder_outputs = encoder_outputs.cuda() if USE_CUDA else encoder_outputs

    loss = 0
    iter_length = min(input_length, max_length)
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input

    # this is the bridge between encoder and decoder. very important
    # this matrix can also be regarded as the features of the input sentence.
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output, encoder_outputs)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input

            loss += criterion(decoder_output, target_variable[di])
            if ni == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length


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



def trainIters(encoder, 
               decoder, 
               n_iters, 
               print_every=1000, 
               plot_every=100, 
               save_every=1000,
               learning_rate=0.01):
    
    start = time.time()
    print("start learning...")
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    
    training_datas = []
    # 随机选取迭代次数的句子组成训练数据集
    for i in range(n_iters):
        var = variable_from_sentence(lang, random.choice(lang.sentences))
        training_datas.append(var)
        
    criterion = nn.NLLLoss()
    # 对于每一个训练数据样本，也就是一句话
    for iter in range(1, n_iters + 1):
        training_data = training_datas[iter - 1]
        input_variable = training_data
        target_variable = training_data

        loss = train(input_variable, 
                     encoder,
                     decoder, 
                     encoder_optimizer, 
                     decoder_optimizer, 
                     criterion)

        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

        if iter % save_every == 0:
            torch.save(encoder, ENCODE_FILE)
            torch.save(decoder, DECODE_FILE)
            print("encoder and decoder saved to files.")

    showPlot(plot_losses)


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import copy

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    input_variable = variable_from_sentence(lang, sentence)
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if USE_CUDA else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                 encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input

    # features of the sentence
    decoder_hidden = encoder_hidden
    sentence_feature = copy.deepcopy(encoder_hidden.data)

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_output, encoder_outputs)
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(lang.index2word[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input

    return (decoded_words, decoder_attentions[:di + 1], \
           sentence_feature, decoder_hidden.data)


def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        sentence = random.choice(lang.sentences)
        print('>', sentence)
        output_words, attentions, _, _ = evaluate(encoder, decoder, sentence)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

# ==========================================
# 训练过程

# 当修改语料库训练文件时，应该删除已经保存的网络文件，使得可以重新生成模型
# 而不是从文件加载训练好的模型


lang = Lang(args.filename)
lang.show_info()
hidden_size = 256

if os.path.isfile(ENCODE_FILE):
    print("load encoder from file")
    encoder1 = torch.load(ENCODE_FILE)
else:
    encoder1 = EncoderRNN(lang.n_words, hidden_size)
if os.path.isfile(DECODE_FILE):
    print("load decoder(attention) from file")
    attn_decoder1 = torch.load(DECODE_FILE)
else:
    attn_decoder1 = AttnDecoderRNN(hidden_size, lang.n_words,
                                   1, dropout_p=0.1)

if USE_CUDA:
    encoder1 = encoder1.cuda()
    attn_decoder1 = attn_decoder1.cuda()

trainIters(encoder1, attn_decoder1, 1000, print_every = 500, save_every=1000)

# 保存网络模型
torch.save(encoder1, ENCODE_FILE)
torch.save(attn_decoder1, DECODE_FILE)

evaluateRandomly(encoder1, attn_decoder1)


# =================================================
# show attention
def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def evaluateAndShowAttention(input_sentence):
    output_words, attentions, _, _ = evaluate(
        encoder1, attn_decoder1, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)

# evaluateAndShowAttention("we should get married .")

# evaluateAndShowAttention("what do announcers do ?")

# evaluateAndShowAttention("You dropped something .")

# evaluateAndShowAttention("You're in danger, Tom .")

# =======================================
# show feature
def show_feature(feature):
    if USE_CUDA:
        feature = feature.cpu()
    feature = feature.numpy().reshape(-1,16)
    print(feature.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(feature, cmap='bone')
    fig.colorbar(cax)
    plt.show()

def evaluateAndShowFeature(input_sentence):
    output_words, attentions, feature, decoder_hidden = evaluate(
        encoder1, attn_decoder1, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    show_feature(feature)
    # show_feature(decoder_hidden)

def compare(v1, v2):
    # 认为v1 v2为PyTorch 张量
    if USE_CUDA:
        v1, v2 = v1.cpu(), v2.cpu()
    v1, v2 = v1.numpy().reshape(1,-1), v2.numpy().reshape(1,-1)
    dis = np.sqrt(np.sum(np.power((v1-v2),2)))
    cos_theta = v1.dot(v2.T)/(np.sqrt(np.sum(v1**2))*np.sqrt(np.sum(v2**2)))
    return dis, cos_theta



def compute_similarity(sentence1, sentence2):
    _,_, feature1, _ = evaluate(encoder1, attn_decoder1, sentence1)
    _,_, feature2, _ = evaluate(encoder1, attn_decoder1, sentence2)
    dis, cos_theta = compare(feature1, feature2)
    print("dis: {}, cos_theta: {}".format(dis, cos_theta))

s1 = "地球 是 方 的"
s2 = "地球 是 太阳系 中 的 一颗 行星"
compute_similarity(s1, s2)

evaluateAndShowFeature(s1)
evaluateAndShowFeature(s2)
