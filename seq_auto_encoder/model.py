from data import *
from torch import nn
import torch
import torch.nn.functional as F
from torch import optim
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence#, masked_cross_entropy
from masked_cross_entropy import *

# self autoencoder
class EncoderRNN(nn.Module):
    # 基于GRU的RNN，作为一个编码器编码序列数据
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        # 词向量自己训练
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if USE_CUDA:
            return result.cuda()
        else:
            return result


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, embedding = None):
        # 使用传入的Embedding来对输入进行向量化，应用于自编码器中，使得
        # encoder 和 decoder使用的是同一套变量
        super(DecoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        if embedding is None:
            self.embedding = nn.Embedding(output_size, hidden_size)
        else:
            self.embedding = embedding            
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if USE_CUDA:
            return result.cuda()
        else:
            return result


class AttnDecoderRNN(nn.Module):
    '''基于注意力机制的RNN解码器'''
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_output, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)))
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]))
        return output, hidden, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if USE_CUDA:
            return result.cuda()
        else:
            return result



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