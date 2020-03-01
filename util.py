from collections import Counter
import numpy as np

import torch


def get_data_from_data_txt(txt_data_path):
    f = open(txt_data_path, 'r')
    data = []
    sentence, tags = [], []
    for line in f.readlines():
        if line != '\n':
            char, tag = line.split()
            sentence.append(char)
            tags.append(tag)
        else:
            data.append((sentence, tags))
            sentence, tags = [], []
    f.close()
    return data

def get_word_to_ix(training_data, min_word_freq=1):
    word_freq = Counter()
    for sentence, _ in training_data:
        word_freq.update(sentence)
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_to_ix = {k: v for v, k in enumerate(words)}
    word_to_ix['<unk>'] = len(word_to_ix)
    word_to_ix['<start>'] = len(word_to_ix)
    word_to_ix['<stop>'] = len(word_to_ix)
    return word_to_ix

def argmax(vec):
    _, idx = torch.max(vec, 1)
    return idx.item()

def prepare_sequence(seq, to_ix, use_unk=False):
    if use_unk:
        unk_ix = to_ix['<unk>']
        idxs = np.array([to_ix.get(w, unk_ix) for w in seq])
        rand_choice = np.random.choice(idxs.shape[0], idxs.shape[0]//100)
        idxs[rand_choice] = unk_ix
        return torch.from_numpy(idxs)
    else:
        unk_ix = to_ix['<unk>']
        idxs = [to_ix.get(w, unk_ix) for w in seq]
        return torch.tensor(idxs)

def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

def get_score_by_label_pred(label_pdf_info, pred_pdf_info):
    type = ('姓名', '出生年月', '性别', '电话', '最高学历', '籍贯', '落户市县', '政治面貌', '毕业院校',
            '工作单位', '工作内容', '职务', '项目名称', '项目责任', '学位', '毕业时间', '工作时间', '项目时间')
    general_type = {'姓名', '出生年月', '性别', '电话', '最高学历', '籍贯', '落户市县', '政治面貌'}
    general_list_type = {'学位', '毕业时间', '工作时间', '项目时间'}
    char_list_type = {'毕业院校', '工作单位', '工作内容', '职务', '项目名称', '项目责任'}
    type_count_for_w = {'毕业院校': 0, '工作单位': 0, '工作内容': 0, '职务': 0, '项目名称': 0, '项目责任': 0}
    type_to_counts = {}
    for t in type:
        type_to_counts[t] = [0, 0, 0]  # [正确总数， 预测总数， 共同数]
    for pdf_name in label_pdf_info.keys():
        label_info = label_pdf_info.get(pdf_name)
        pred_info = pred_pdf_info.get(pdf_name, {})
        for t in label_info:
            if t in general_type:
                type_to_counts[t][0] += 1
                if t in pred_info:
                    type_to_counts[t][1] += 1
                    if label_info[t] == pred_info[t]:
                        type_to_counts[t][2] += 1
            elif t in general_list_type:
                type_to_counts[t][0] += len(label_info[t])
                if t in pred_info:
                    type_to_counts[t][1] += len(pred_info[t])
                    commond_idx = set()
                    for l in label_info[t]:
                        for i, p in enumerate(pred_info[t]):
                            if l == p and i not in commond_idx:
                                commond_idx.add(i)
                                break
                    type_to_counts[t][2] += len(commond_idx)
            elif t in char_list_type:
                type_count_for_w[t] += len(label_info[t])
                for l in label_info[t]:
                    type_to_counts[t][0] += len(l)
                if t in pred_info:
                    for p in pred_info[t]:
                        type_to_counts[t][1] += len(p)
                    type_to_counts[t][2] += get_common_char_count_by_l_p(label_info[t], pred_info[t])
    print(type_to_counts)
    sum_count = 0
    for type in type_to_counts:
        if type not in type_count_for_w:
            cnt = type_to_counts[type][0]
            type_count_for_w[type] = cnt
        else:
            cnt = type_count_for_w[type]
        sum_count += cnt

    final_score = 0
    for type, counts in type_to_counts.items():
        label_count, pred_count, commond_count = counts
        if commond_count == 0:
            score = 0
        else:
            score = get_f1_by_p_r_w(commond_count / label_count,
                                    commond_count / pred_count, type_count_for_w[type] / sum_count)
        print(type, score, '/', type_count_for_w[type] / sum_count)
        final_score += score
    return final_score

def get_common_char_count_by_l_p(label, pred):
    sum_common_len = 0
    common_idx = set()
    for l in label:
        cur_max_length, cur_max_idx = -1, -1
        for i, p in enumerate(pred):
            if i not in common_idx:
                cur_l = get_common_str_len(l, p)
                if cur_l > cur_max_length:
                    cur_max_length = cur_l
                    cur_max_idx = i
        if cur_max_length != -1:
            common_idx.add(cur_max_idx)
            sum_common_len += cur_max_length
    return sum_common_len

def get_common_str_len(str1, str2):
    matrix = []
    xmax = 0
    xindex = 0
    for i, x in enumerate(str2):
        matrix.append([])
        for j, y in enumerate(str1):
            if x != y:
                matrix[i].append(0)
            else:
                if i == 0 or j == 0:
                    matrix[i].append(1)
                else:
                    matrix[i].append(matrix[i - 1][j - 1] + 1)
                if matrix[i][j] > xmax:
                    xmax = matrix[i][j]
                    xindex = j
                    xindex += 1
    return xmax


def get_f1_by_p_r_w(p, r, w):
    return w * (2 * p * r) / (p + r)

def write_info_by_ix(ix, content, ix_to_tag):
    tags = [ix_to_tag[i] for i in ix]
    fine_new = True
    start_idxs, end_idxs, pred_tags = [], [], []
    for i, tag in enumerate(tags):
        if fine_new:
            if tag[0] == 'b' or tag[0] == 'c':
                start_idxs.append(i)
                fine_new = False
                pred_tags.append(tag[2:])
        else:
            if tag != 'i-' + pred_tags[-1]:
                end_idxs.append(i)
                if tag[0] == 'b' or tag[0] == 'c':
                    start_idxs.append(i)
                    pred_tags.append(tag[2:])
                else:
                    fine_new = True
    if len(start_idxs) != len(end_idxs):
        end_idxs.append(len(tags))
    tag_to_cn = {'name': '姓名', 'bir': '出生年月', 'gend': '性别', 'tel': '电话', 'acad': '最高学历',
                 'nati': '籍贯', 'live': '落户市县', 'poli': '政治面貌', 'unv': '毕业院校', 'comp': '工作单位',
                 'work': '工作内容', 'post': '职务', 'proj': '项目名称', 'resp': '项目责任', 'degr': '学位',
                 'grti': '毕业时间', 'woti': '工作时间', 'prti': '项目时间'}
    single_cn_tag = {'姓名', '出生年月', '性别', '电话', '最高学历', '籍贯', '落户市县', '政治面貌'}
    info = {}
    for i, p_tag in enumerate(pred_tags):
        cn_tag = tag_to_cn[p_tag]
        if cn_tag in single_cn_tag:
            if cn_tag not in info:
                info[cn_tag] = content[start_idxs[i]: end_idxs[i]]
        else:
            if cn_tag not in info:
                info[cn_tag] = []
            info[cn_tag].append(content[start_idxs[i]: end_idxs[i]])
    return info

def write_info_by_ix_plus(ix, content, ix_to_tag):
    tags = [ix_to_tag[i] for i in ix]
    fine_new = True
    start_idxs, end_idxs, pred_tags, c_tag_idxs = [], [], [], set()
    for i, tag in enumerate(tags):
        if fine_new:
            if tag[0] == 'b' or tag[0] == 'c':
                start_idxs.append(i)
                fine_new = False
                pred_tags.append(tag[2:])
                if tag[0] == 'c':
                    c_tag_idxs.add(len(pred_tags)-1)
        else:
            if tag != 'i-' + pred_tags[-1]:
                end_idxs.append(i)
                if tag[0] == 'b' or tag[0] == 'c':
                    start_idxs.append(i)
                    pred_tags.append(tag[2:])
                    if tag[0] == 'c':
                        c_tag_idxs.add(len(pred_tags)-1)
                else:
                    fine_new = True
    if len(start_idxs) != len(end_idxs):
        end_idxs.append(len(tags))
    tag_to_cn = {'name': '姓名', 'bir': '出生年月', 'gend': '性别', 'tel': '电话', 'acad': '最高学历',
                 'nati': '籍贯', 'live': '落户市县', 'poli': '政治面貌', 'unv': '毕业院校', 'comp': '工作单位',
                 'work': '工作内容', 'post': '职务', 'proj': '项目名称', 'resp': '项目责任', 'degr': '学位',
                 'grti': '毕业时间', 'woti': '工作时间', 'prti': '项目时间'}
    single_cn_tag = {'姓名', '出生年月', '性别', '电话', '最高学历', '籍贯', '落户市县', '政治面貌'}
    info = {}
    for i, p_tag in enumerate(pred_tags):
        cn_tag = tag_to_cn[p_tag]
        if cn_tag in single_cn_tag:
            if cn_tag not in info:
                info[cn_tag] = content[start_idxs[i]: end_idxs[i]]
            elif i in c_tag_idxs:
                info[cn_tag] += content[start_idxs[i]: end_idxs[i]]
        else:
            if cn_tag not in info:
                info[cn_tag] = []
            if i not in c_tag_idxs:
                info[cn_tag].append(content[start_idxs[i]: end_idxs[i]])
            else:
                if len(info[cn_tag]) > 0:
                    info[cn_tag][-1] += content[start_idxs[i]: end_idxs[i]]
    return info

def unk_to_part_of_word(sentence):
    length = len(sentence)


if __name__ == '__main__':
    # len = get_common_str_len('sadg', 'sadgh')
    # print(len)
    l = ['asdf', 'qwer', 'sadg', 'sadb']
    p = ['sdfgh', 'qwt', 'sadf', 'asdf']
    common_char = get_common_char_count_by_l_p(l, p)
    print(common_char)