# s = "wef:fw：few"
#
# print(s.split(':'))
#
# import os
# import re
#
# TRAIN_PDF_PATH = '/home/agwave/Data/resume/resume_train_20200121/pdf'
# ORIG_TXT_PATH = '/home/agwave/Data/resume/resume_train_20200121/orig_txt/f2be69555427.txt'
#
#
#
# def read_txt(path):
#     info = []
#     with open(path, 'r') as f:
#         for line in f.readlines():
#             clean_line = line.strip()
#             words = re.split('[:：]', clean_line)
#             for word in words:
#                 w = word.strip()
#                 if w != '':
#                     info.append(w)
#     return info
# import numpy as np
# a = np.array([[1, 2], [3, 4]])
# for i in a:
#     print(i.shape)
# content = 'asdfewfwefewff'
# s = 'fewff'
# # print(content.find(s))
# a, b = [1, 2]
# print(a, b)
# f = open('/home/agwave/Data/resume/resume_train_20200121/final_data_simple.txt', 'r')
# data = []
# sentence, tags = [], []
# for line in f.readlines():
#     if line != '\n':
#         char, tag = line.split()
#         sentence.append(char)
#         tags.append(tag)
#         print(char, tag)
#     else:
#         data.append((sentence, tags))
#         sentence, tags = [], []
# f.close()

# 测试LSTM多batch训练
# import torch
# import torch.nn as nn
#
# lstm = nn.LSTM(input_size=5, hidden_size=5, num_layers=1, bidirectional=True)
# length = torch.tensor([6, 9, 7])
# sentences = [['we', 'are', 'good', 'peopel', 'so', 'happy'],
#             ['I', 'like', 'play', 'compute', 'game', 'it', 'let', 'me', 'happy'],
#             ['she', 'love', 'sing', 'song', 'listen', 'really', 'good']]
# word_to_ix = {'<pad>': 0}
# for sentence in sentences:
#     for word in sentence:
#         if word not in word_to_ix:
#             word_to_ix[word] = len(word_to_ix)
#
#
# def prepare_sequence(seq, to_ix):
#     idxs = [to_ix[w] for w in seq]
#     return torch.tensor(idxs, dtype=torch.long)
# for sen in sentences:
#     sen += ['<pad>'] * (9 - len(sen))
#
# input = []
# for sen in sentences:
#     enc_cap = prepare_sequence(sen, word_to_ix)
#     input.append(torch.tensor(enc_cap).clone().detach())
#
# input = torch.stack(input, 0).permute(1, 0)
# print(input.size())
# embedding = nn.Embedding(len(word_to_ix), 5)
# input = embedding(input)
# print(input.size())
# new_length, sort_idx = torch.sort(length, descending=True)
#
# _, unsort_idx = torch.sort(sort_idx)
#
# new_input = input.index_select(1, sort_idx)
#
# final_input = torch.nn.utils.rnn.pack_padded_sequence(new_input, new_length)
# c, _ = lstm(final_input)
#
# c, _ = torch.nn.utils.rnn.pad_packed_sequence(c)
# print(c.size())
# nc = c.index_select(1, unsort_idx)
# print(nc.size())

# Counter() 测试
# from collections import Counter
#
# word_freq = Counter()
# a = ['安慰', 'wef', 'ewf']
# word_freq.update(a)
# print(word_freq['wef'])


"""data_process 无用垃圾放置处

def str_segmentation_to_word(str):
    jieba.enable_paddle()
    words = list(jieba.cut(str, use_paddle=True))
    return words


def all_pdf_words_to_txt(pdf_dir, corpus_path):
    jieba.enable_paddle()
    pdf_path_list = os.listdir(pdf_dir)
    corpus = ''
    for pdf_name in pdf_path_list:
        pdf_path = os.path.join(pdf_dir, pdf_name)
        pdf_str = get_str_from_pdf(pdf_path)
        pdf_str_cut = jieba.cut(pdf_str, use_paddle=True)
        final_str = ' '.join(pdf_str_cut)
        corpus += final_str
        corpus += ' '
    with open(os.path.join(corpus_path), 'w+', encoding='utf-8') as f:
        f.write(corpus)

def build_word2vec_model(corpus_path):
    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
    sentences = word2vec.Text8Corpus(corpus_path)
    model = word2vec.Word2Vec(sentences, size=100, hs=1, min_count=1)
    return model


def model_test(model):
    try:
        sim3 = model.most_similar('管理', topn=20)
        for key in sim3:
            print(key[0], key[1])
    except:
        print('error')

"""

"""gen_json.py 垃圾放置

def write_info(content):
    info = {}
    info['姓名'] = ext_general_field(content, '名')
    info['籍贯'] = ext_general_field(content, '籍贯')
    info['出生日期'] = ext_general_field(content, '出生')
    info['电话'] = ext_general_field(content, '电话')
    info['最高学历'] = ext_general_field(content, '学历')
    info['落户市县'] = ext_general_field(content, '落户')
    info['政治面貌'] = ext_general_field(content, '面貌')
    try:
        if info['姓名'] == '':
            info['姓名'] = content[0]
        if info['出生日期'] == '':
            for c in content:
                if c[:2].isdigit() == True and c.isdigit() == False:
                    info['出生日期'] = c
                    break
        if info['电话'] == '':
            for c in content:
                if c.isdigit():
                    info['电话'] = c
                    break
        if info['最高学历'] == '':
            for c in content:
                if '士' in c:
                    info['最高学历'] = c
                    break
        if info['政治面貌'] == '':
            for c in content:
                if '员' in c:
                    info['政治面貌'] = c
                    break
    except Exception as e:
        print(e)
    remove_key = []
    for key, value in info.items():
        if info[key] == '':
            remove_key.append(key)
    for k in remove_key:
        info.pop(k)
    return info


def ext_general_field(content, part):
    con_len = len(content)
    for i, c in enumerate(content):
        if part in c and i < con_len-1:
           return content[i+1]
    return ''
    
def pdf2strlist(pdf_path):
    strlist = []
    content = ''
    if pdf_path.endswith('.pdf'):
        rsrcmgr = PDFResourceManager(caching=True)
        laparams = LAParams()
        retstr = StringIO()
        device = TextConverter(rsrcmgr, retstr, laparams=laparams)
        with open(pdf_path, 'rb') as fp:
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            for page in PDFPage.get_pages(fp, pagenos=set()):
                page.rotate = page.rotate % 360
                interpreter.process_page(page)
        device.close()
        content = retstr.getvalue()
    if content != '':
        words = content.strip().replace('\n', '').split()
        for word in words:
            word = re.split('[:：]', word)
            for w in word:
                strlist.append(w)
        strlist = list(filter(lambda x: x, strlist))
    return strlist
"""

"""测试get_word_to_ix
from util import prepare_sequence
from util import get_word_to_ix

training_data = [[[1, 2, 3], [3, 1, 2]]]
a = [1, 2, 3, 4, 6]
word_to_ix = get_word_to_ix(training_data, min_word_freq=0)
b = prepare_sequence(a, word_to_ix)
print(b)
"""
#
# a = 'wfwefwef'
# b = list(a)
# print(a)
# print(b)

# a = 'wefwefdscvtyh'
# print(a[2:13])

# type = {'姓名', '出生年月', '性别', '电话', '最高学历', '籍贯', '落户市县', '政治面貌', '毕业院校',
#         '工作单位', '工作内容', '职务', '项目名称', '项目责任', '学位', '毕业时间', '工作时间', '项目时间'}
# type_to_counts = {}
# for t in type:
#     type_to_counts[t] = [0, 0, 0, 0]  # [正确总数， 预测总数， 召回数， 预测准确数]
# for c in {'姓名'}:
#     type_to_counts[c][2] += 1
# print(type_to_counts)
# a = {0: 11}
# b = a.get(1)
# print(b)
# a = {1, 2}
# print(len(a))
# def longestCommonSubsequence(str1, str2) -> int:
#     m, n = len(str1), len(str2)
#     # 构建 DP table 和 base case
#     dp = [[0] * (n + 1) for _ in range(m + 1)]
#     # 进行状态转移
#     for i in range(1, m + 1):
#         for j in range(1, n + 1):
#             if str1[i - 1] == str2[j - 1]:
#                 # 找到一个 lcs 中的字符
#                 dp[i][j] = 1 + dp[i - 1][j - 1]
#             else:
#                 dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
#
#     return dp[-1][-1]
# com = longestCommonSubsequence('asdfgh', 'dfgoui')
# print(com)

# import json
# import os
# def get_sample_real_json(sample_pdf_dir, real_json_path):
#     name_to_info = {}
#     with open(real_json_path, 'r') as j:
#         label_info = json.load(j)
#     paths = os.listdir(sample_pdf_dir)
#     names = []
#     for p in paths:
#         pdf_name = p[:-4]
#         names.append(pdf_name)
#     for name in names:
#         if name in label_info:
#             name_to_info[name] = label_info[name]
#     with open('/home/agwave/Data/resume/resume_train_20200121/pdf_sample_real.json', 'w') as j:
#         json.dump(name_to_info, j, ensure_ascii=False)
#
# sample_pdf_dir = '/home/agwave/Data/resume/resume_train_20200121/pdf_simple/'
# real_json_path = '/home/agwave/Data/resume/resume_train_20200121/own_train_data.json'
# get_sample_real_json(sample_pdf_dir, real_json_path)

# import numpy as np
# import torch
#
# A = np.array([3, 5, 6])
# B= np.arange(2, 10)
# B[A] = 0
# C = torch.tensor(A, dtype=torch.long)
# print(C)
# print(C.type())

# import torch
#
# A = torch.tensor([36], dtype=torch.long)
# B = torch.tensor([2, 4, 2,54, 657], dtype=torch.long)
# C = torch.cat([A, B])
# print(A)
# print(B.size())
# print(C)

# import json
# with open('word_to_ix_add_unk_0219.json') as j:
#     word_to_ix = json.load(j)
# print(len(word_to_ix))

# 测试torch.max
# A = torch.tensor([[1, 2, 3], [4, 5, 6]])

# txt_path = 'wrong_pdf.txt'
# tags = set()
# with open(txt_path, 'r') as f:
#     for l in f.readlines():
#         line = l.split()
#         if len(line) == 4:
#             if line[-1] not in tags:
#                 tags.add(line[-1])
# print(tags)
# {'live', 'proj', 'woti', 'post', 'unv', 'nati', 'poli', 'prti', 'comp'}
# a = 'abfdwfbffew'
# b = 'bf'
# a = a.replace(b, '')
# print(a)

# 打乱列表
import random

A = [1, 2, 3, 4]
random.shuffle(A)
for p in A[:2]:
    print(p)