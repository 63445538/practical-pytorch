

import unicodedata
import re
import random
import torch
from torch.autograd import Variable


# loading data files
# indexing words

# 一些标志符号(词)
PAD_token = 0   # 补齐用标志
SOS_token = 1   # 句子起始
EOS_token = 2   # 句子结束

USE_CUDA = True
if USE_CUDA:
    print("Use CUDA")

MIN_COUNT = 5

MIN_LENGTH = 3
MAX_LENGTH = 25
NORMALIZE = False

# reading and decoding files
# Turn a Unicode string to plain ASCII, thanks to：
# http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s): # 完成从Unicode到AsciII编码的转换
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalize_string(s): # 去掉一些不是字母的字符
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([,.!?])", r" \1 ", s) # 在指定字符前后增加空格
    s = re.sub(r"[^a-zA-Z,.!?]+", r" ", s) # 用空格去掉一些非字母即指定标点的字符。
    s = re.sub(r"\s+", r" ", s).strip() # 去掉首尾空格
    return s

def filter_sentences(sentences): # 只选用一定长度的句子
    filtered = []
    for sentence in sentences:
        if len(sentence) >= MIN_LENGTH and len(sentence) <= MAX_LENGTH:
                filtered.append(sentence)
    return filtered


# Pad 是句子长度不到最大长度时的占位符吗？要把seq补足长度？
def pad_seq(seq, max_length):
    seq += [PAD_token for i in range(max_length - len(seq))]
    return seq

# 根据句子组成找到句子里所有词语对应的向量索引
def indexes_from_sentence(lang, sentence):
    return lang.indexes(sentence)

# 将句子转为一个PyTorch变量
def variable_from_sentence(lang, sentence):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1,1))
    if USE_CUDA:
        return result.cuda()
    else:
        return result

# 输出时间辅助信息的方法
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

class Lang(object):
    def __init__(self, file_name=None, name=None):
        self.name = name
        self.trimmed = False    # 是否去掉了一些不常用的词
        self.word2index = {}    # 词-> 索引
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"} # 索引 -> 词
        self.n_words = 3 # 默认收录到词典里的在训练库里出现的最少次数
        self.sentences = [] # 收录的句子
        # self.min_count = MIN_COUNT
        if file_name is not None:
            self.load(file_name)


    def index_words(self, sentence):    # 从句子收录词语到字典
        for word in sentence.split(' '):
            self.index_word(word)

    def index_word(self, word): # 收录一个词到词典
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def load(self, file_name):# 从文件加载语料库
        lines = open(file_name).read().strip().split('\n')
        for sentence in lines:
            if NORMALIZE:
                sentence = normalize_string(sentence)
            self.sentences.append(sentence)
            self.index_words(sentence)

    # 词在语料库里出现的次数低于指定的次数时将把该词剔除
    def trim(self, min_count):
        if self.trimmed: return
        self.trimmed = True
        
        keep_words = []
        
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words %s / %s = %.4f' % (
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.n_words = 3 # Count default tokens

        for word in keep_words:
            self.index_word(word)

    def indexes(self, sentence): # 根据lang返回一个句子的张量
    # 碰到词典里没有的word，self.word2index[word]会将其添加至词典中。
        indexes = []
        if NORMALIZE:
            sentence = normalize_string(sentence)
        for word in sentence.split(' '):
            if self.word2index.get(word) is None:# 不存在该单词
                self.index_word(word)
            indexes.append(self.word2index[word])
        return indexes + [EOS_token]

    def random_batch(self, batch_size):
        # batch_size=1时即不会出现使用<PAD>补足位置的问题
        # 该方法好像没有被调用过
        input_seqs = []
        for i in range(batch_size):
            sentence = random.choice(self.sentences)
            input_seqs.append(self.indexes(sentence))
        
        input_lengths = [len(s) for s in input_seqs]
        input_padded = [pad_seq(s, max(input_lengths)) for s in input_seqs]
        input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)

        if USE_CUDA:
            input_var = input_var.cuda()

        return input_var, input_lengths

    def show_info(self):

        print("n_words:{0}, n_sentences:{1}".format(self.n_words,
                                                    len(self.sentences)))

def test():
    lang = Lang(file_name="chinese.txt")
    print("n_words:{0}, n_sentences:{1}".format(self.n_words,
                                                len(self.sentences)))

if __name__ == "__main__":
    test()