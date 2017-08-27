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
argparser.add_argument('filename', type=str, default="data/10-common.txt")

args = argparser.parse_args()
file, ext = os.path.splitext(args.filename)

ENCODE_FILE = file + "_encoder_auto.pt"
DECODE_FILE = file + "_decoder_auto.pt"


lang = Lang(args.filename)
lang.show_info()
hidden_size = 256

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


# ==========================================
# 训练过程

# 当修改语料库训练文件时，应该删除已经保存的网络文件，使得可以重新生成模型
# 而不是从文件加载训练好的模型


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



s1 = "地球 是 方 的"
s2 = "地球 是 太阳系 中 的 一颗 行星"
compute_similarity(s1, s2)

evaluateAndShowFeature(s1)
evaluateAndShowFeature(s2)
