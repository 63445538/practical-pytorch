import unicodedata
import re
import random
import torch
from torch.autograd import Variable


# loading data files
# indexing words

# 一些标志符号(词)
PAD_token = 0
SOS_token = 1   # 句子起始
EOS_token = 2   # 句子结束

USE_CUDA = False

class Lang:
    def __init__(self, name):
        self.name = name
        self.trimmed = False    # 是否去掉了一些不常用的词
        self.word2index = {}    # 词-> 索引
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"} # 索引 -> 词
        self.n_words = 3 # 默认收录到词典里的在训练库里出现的最少次数

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

def read_langs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
#     filename = '../data/%s-%s.txt' % (lang1, lang2)
    filename = '../%s-%s.txt' % (lang1, lang2)
    lines = open(filename).read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

MIN_LENGTH = 3
MAX_LENGTH = 25

def filter_pairs(pairs): # 去掉任何长度不符合要求的句子对
    filtered_pairs = []
    for pair in pairs:
        if len(pair[0]) >= MIN_LENGTH and len(pair[0]) <= MAX_LENGTH \
            and len(pair[1]) >= MIN_LENGTH and len(pair[1]) <= MAX_LENGTH:
                filtered_pairs.append(pair)
    return filtered_pairs

def prepare_data(lang1_name, lang2_name, reverse=False):
    input_lang, output_lang, pairs = read_langs(lang1_name, lang2_name, reverse)
    print("Read %d sentence pairs" % len(pairs))
    
    pairs = filter_pairs(pairs)
    print("Filtered to %d pairs" % len(pairs))
    
    print("Indexing words...")
    for pair in pairs:
        input_lang.index_words(pair[0])
        output_lang.index_words(pair[1])
    
    print('Indexed %d words in input language, %d words in output' % (input_lang.n_words, output_lang.n_words))
    return input_lang, output_lang, pairs

input_lang, output_lang, pairs = prepare_data('eng', 'fra', True)


MIN_COUNT = 5 # 出现次数低于5次的次将被扔掉

input_lang.trim(MIN_COUNT)
output_lang.trim(MIN_COUNT)

keep_pairs = []

# 去掉一些句子对，在这些句子对中存在被剔除出任一词典的词。
for pair in pairs:
    input_sentence = pair[0]
    output_sentence = pair[1]
    keep_input = True
    keep_output = True
    
    for word in input_sentence.split(' '):
        if word not in input_lang.word2index:
            keep_input = False
            break

    for word in output_sentence.split(' '):
        if word not in output_lang.word2index:
            keep_output = False
            break

    # Remove if pair doesn't match input and output conditions
    if keep_input and keep_output:
        keep_pairs.append(pair)


print("Trimmed from %d pairs to %d, %.4f of total" %\
    (len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
pairs = keep_pairs


# Turning training data into Tensors

# Return a list of indexes, one for each word in the sentence, plus EOS
# 先得到一个句子在某一语种词典中的索引
def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')] + [EOS_token]


# Pad a with the PAD symbol
# Pad 是句子长度不到最大长度时的占位符吗？要把seq补足长度？
def pad_seq(seq, max_length):
    seq += [PAD_token for i in range(max_length - len(seq))]
    return seq

def random_batch(batch_size):
    input_seqs = []
    target_seqs = []

    # Choose random pairs
    for i in range(batch_size):
        pair = random.choice(pairs)
        input_seqs.append(indexes_from_sentence(input_lang, pair[0]))
        target_seqs.append(indexes_from_sentence(output_lang, pair[1]))

    # Zip into pairs, sort by length (descending), unzip
    seq_pairs = sorted(zip(input_seqs, target_seqs), key=lambda p: len(p[0]), reverse=True)
    input_seqs, target_seqs = zip(*seq_pairs)
    
    # For input and target sequences, get array of lengths and pad with 0s to max length
    input_lengths = [len(s) for s in input_seqs]
    input_padded = [pad_seq(s, max(input_lengths)) for s in input_seqs]
    target_lengths = [len(s) for s in target_seqs]
    target_padded = [pad_seq(s, max(target_lengths)) for s in target_seqs]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)
    target_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1)
    
    if USE_CUDA:
        input_var = input_var.cuda()
        target_var = target_var.cuda()
        
    return input_var, input_lengths, target_var, target_lengths


input_var, input_length, target_var, target_length = random_batch(2)
print("{}, {}".format(input_length, target_length))
print(input_var)
print(target_var)
