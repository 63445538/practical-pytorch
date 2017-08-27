# https://github.com/spro/practical-pytorch

import unidecode
import string
import random
import time
import math
import torch
from torch.autograd import Variable
import re
# Reading and un-unicode-encoding data
def convert_file(filename, new_file=None):
    r_file = open(filename, "r")
    lines = r_file.read().split("\n")
    if new_file is None:
        new_file = filename+"1"
    w_file = open(new_file,"w")
    for line in lines:
        newline=re.sub(r"[0-9a-zA-Z,：.!?]+", r"", line)
        w_file.write(newline+"\n")
    w_file.close()
    

# 根据文本来获取所有的字符集
class Lang(object):
    def __init__(self, file_name=None, name=None):
        self.name = name
        self.trimmed = False    # 是否去掉了一些不常用的词
        self.char2index = {'\n':0}    # 词-> 索引
        self.char2count = {}
        self.index2char = {0:'\n'} # 索引 -> 词
        self.n_chars = 1 # 默认收录到词典里的在训练库里出现的最少次数
        # self.min_count = MIN_COUNT
        if file_name is not None:
            self.load(file_name)


    def index_chars(self, line):    # 从句子收录词语到字典
        for char in line:
            self.index_char(char)

    def index_char(self, char): # 收录一个词到词典
        if char not in self.char2index:
            self.char2index[char] = self.n_chars
            self.char2count[char] = 1
            self.index2char[self.n_chars] = char
            self.n_chars += 1
        else:
            self.char2count[char] += 1

    def load(self, file_name):# 从文件加载语料库
        lines = open(file_name).read().strip().split('\n')
        for line in lines:
            self.index_chars(line)

    # 词在语料库里出现的次数低于指定的次数时将把该词剔除
    def trim(self, min_count):
        if self.trimmed: return
        self.trimmed = True
        
        keep_chars = []
        
        for k, v in self.char2count.items():
            if v >= min_count:
                keep_chars.append(k)

        print('keep_chars %s / %s = %.4f' % (
            len(keep_chars), len(self.char2index), len(keep_chars) / len(self.char2index)
        ))

        # Reinitialize dictionaries
        self.char2index = {'\n':0}    # 词-> 索引
        self.char2count = {}
        self.index2char = {0:'\n'} # 索引 -> 词
        self.n_chars = 1 # 默认收录到词典里的在训练库里出现的最少次数

        for char in keep_chars:
            self.index_char(char)

    def indexes(self, line): # 根据lang返回一个句子的张量
    # 碰到词典里没有的char，self.char2index[char]会将其添加至词典中。
        indexes = []
        for char in line:
            if self.char2index.get(char) is None:# 不存在该单词
                self.index_char(char)
            indexes.append(self.char2index[char])
        return indexes

    def show_info(self):

        print("收录{0}个字（符）".format(self.n_chars))


def read_file(filename):
    lang = Lang(filename)
    #lang.trim(5)
    global n_characters
    n_characters = len(lang.char2index)
    #print("In read file")
    #print(n_characters)
    file = open(filename).read()
    # file = unidecode.unidecode(open(filename).read())
    #print(len(file))
    return file, len(file), lang

# Turning a string into a tensor

def char_tensor(lang, string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = lang.char2index[string[c]]
    return Variable(tensor)

# Readable time elapsed

def time_since(since):
    s = time.time() - float(since)
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
