from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage

import shutil
import os
import json
from io import StringIO
import time

import logging

sample1_pdf_path = '/home/agwave/Data/resume/resume_train_20200121/pdf/0ac186229aa0.pdf'
sample2_pdf_path = '/home/agwave/Data/resume/resume_train_20200121/pdf/a0a9e29f84aa.pdf'

PDF_DIR = '/home/agwave/Data/resume/resume_train_20200121/pdf/'
PDF_TEST_DIR = '/home/agwave/Data/resume/resume_train_20200121/pdf_simple/'
CORPUS_PATH = '/home/agwave/Data/resume/resume_train_20200121/corpus.txt'

MODEL_SAVE_PATH = '/home/agwave/Data/resume/resume_train_20200121/word2vec.model'

OLD_JSON_PATH = '/home/agwave/Data/resume/[new] train_data_20200207.json'
NEW_JOSN_PATH = '/home/agwave/Data/resume/own_train_data_20200207.json'

FINAL_DATA_PATH = '/home/agwave/Data/resume/word_to_tag.txt'


def get_str_from_pdf(pdf_path):
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
    words = content.strip().replace('\n', '').split()
    ret = ''.join(words)
    ret = ret.replace('简历来自：BOSS直聘', '')
    return ret


def build_json_from_old_one(old_json_path, new_json_path):
    old_json = open(old_json_path, 'r')
    old_data = json.load(old_json)
    old_json.close()
    new_data = {}
    for name, info in old_data.items():
        new_info = {}
        for key, value in info.items():
            if key not in {'教育经历', '工作经历', '项目经历'}:
                new_info[key] = value
            else:
                for d in value:
                    for k, v in d.items():
                        if k not in new_info:
                            new_info[k] = []
                        new_info[k].append(v)
        new_data[name] = new_info
    json.dump(new_data, open(new_json_path, 'w', encoding='utf-8'), ensure_ascii=False)


def tagging2txt(pdf_dir, tag_file_path, txt_path):
    logging.basicConfig(level=logging.INFO, filename='supporting_document/log.txt', format='%(message)s')
    tag_file = open(tag_file_path, 'r')
    if os.path.exists(txt_path):
        os.remove(txt_path)
    txt_file = open(txt_path, 'w+')
    tags = json.load(tag_file)
    pdf_path_list = os.listdir(pdf_dir)
    tag_to_biname = {'姓名': 'name', '出生年月': 'bir', '性别': 'gend', '电话': 'tel', '最高学历': 'acad',
                     '籍贯': 'nati', '落户市县': 'live', '政治面貌': 'poli', '毕业院校': 'unv', '工作单位': 'comp',
                     '工作内容': 'work', '职务': 'post', '项目名称': 'proj', '项目责任': 'resp', '学位': 'degr',
                     '毕业时间': 'grti', '工作时间': 'woti', '项目时间': 'prti'}

    for p in pdf_path_list:
        if p.endswith('.pdf'):
            pdf_name = p[:-4]
            pdf_path = os.path.join(pdf_dir, p)
            content = get_str_from_pdf(pdf_path)
            con_to_tag = ['o'] * len(content)
            for tag, sign in tags[pdf_name].items():
                if isinstance(sign, str):
                    idx = content.find(sign)
                    if idx != -1:
                        con_to_tag[idx] = 'b-'+tag_to_biname[tag]
                        for i in range(1, len(sign)):
                            con_to_tag[idx+i] = 'i-'+tag_to_biname[tag]
                    else:
                        logging.info(pdf_name + ' ' + sign + ' ' + tag_to_biname[tag])
                else:
                    for s in sign:
                        idx = content.find(s)
                        if idx != -1:
                            con_to_tag[idx] = 'b-'+tag_to_biname[tag]
                            for i in range(1, len(s)):
                                con_to_tag[idx+i] = 'i-'+tag_to_biname[tag]
                        else:
                            logging.info(pdf_name + ' ' + s + ' ' + tag_to_biname[tag])
            assert len(content) == len(con_to_tag)

            for i in range(len(content)):
                txt_file.write(content[i] + ' ' + con_to_tag[i] + '\n')
            txt_file.write('\n')
    tag_file.close()
    txt_file.close()

def get_dict_for_better_tagging_by_wrong_txt(wrong_txt_path):
    ret = {}
    with open(wrong_txt_path, 'r') as f:
        for result in f.readlines():
            line = result.split()
            if len(line) == 4:
                if line[0] not in ret:
                    ret[line[0]] = []
                ret[line[0]].append(line[1:])
    return ret

def tagging2txt_with_c(pdf_dir, tag_file_path, txt_path):
    logging.basicConfig(level=logging.INFO, filename='supporting_document/log.txt', format='%(message)s')
    with open(tag_file_path, 'r') as j:
        tags = json.load(j)
    if os.path.exists(txt_path):
        os.remove(txt_path)
    txt_file = open(txt_path, 'w+')
    pdf_path_list = os.listdir(pdf_dir)
    tag_to_biname = {'姓名': 'name', '出生年月': 'bir', '性别': 'gend', '电话': 'tel', '最高学历': 'acad',
                     '籍贯': 'nati', '落户市县': 'live', '政治面貌': 'poli', '毕业院校': 'unv', '工作单位': 'comp',
                     '工作内容': 'work', '职务': 'post', '项目名称': 'proj', '项目责任': 'resp', '学位': 'degr',
                     '毕业时间': 'grti', '工作时间': 'woti', '项目时间': 'prti'}
    pdf_name_to_tags = get_dict_for_better_tagging_by_wrong_txt('supporting_document/wrong_pdf.txt')
    for p in pdf_path_list:
        if p.endswith('.pdf'):
            pdf_name = p[:-4]
            pdf_path = os.path.join(pdf_dir, p)
            content = get_str_from_pdf(pdf_path)
            con_to_tag = ['o'] * len(content)
            for tag, sign in tags[pdf_name].items():
                if isinstance(sign, str):
                    idx = content.find(sign)
                    if idx != -1:
                        con_to_tag[idx] = 'b-' + tag_to_biname[tag]
                        for i in range(1, len(sign)):
                            con_to_tag[idx + i] = 'i-' + tag_to_biname[tag]
                    else:
                        has_wrong = True
                        if pdf_name in pdf_name_to_tags:
                            for front, behind, wrong_tag in pdf_name_to_tags[pdf_name]:
                                if front + behind == sign:
                                    f_i = content.find(front)
                                    if f_i != -1:
                                        b_i = content.find(behind, f_i+len(front))
                                        if b_i != -1:
                                            con_to_tag[f_i] = 'b-' + wrong_tag
                                            con_to_tag[b_i] = 'c-' + wrong_tag
                                            for i in range(1, len(front)):
                                                con_to_tag[f_i+i] = 'i-' + wrong_tag
                                            for i in range(1, len(behind)):
                                                con_to_tag[b_i+i] = 'i-' + wrong_tag
                                            has_wrong = False
                                    break
                        if has_wrong:
                            logging.info(pdf_name + ' ' + sign + ' ' + tag_to_biname[tag])
                else:
                    for s in sign:
                        idx = content.find(s)
                        if idx != -1:
                            con_to_tag[idx] = 'b-' + tag_to_biname[tag]
                            for i in range(1, len(s)):
                                con_to_tag[idx + i] = 'i-' + tag_to_biname[tag]
                        else:
                            has_wrong = True
                            if pdf_name in pdf_name_to_tags:
                                for front, behind, wrong_tag in pdf_name_to_tags[pdf_name]:
                                    if front + behind == s:
                                        f_i = content.find(front)
                                        if f_i != -1:
                                            b_i = content.find(behind, f_i + len(front))
                                            if b_i != -1:
                                                con_to_tag[f_i] = 'b-' + wrong_tag
                                                con_to_tag[b_i] = 'c-' + wrong_tag
                                                for i in range(1, len(front)):
                                                    con_to_tag[f_i + i] = 'i-' + wrong_tag
                                                for i in range(1, len(behind)):
                                                    con_to_tag[b_i + i] = 'i-' + wrong_tag
                                                has_wrong = False
                                        break
                            if has_wrong:
                                logging.info(pdf_name + ' ' + s + ' ' + tag_to_biname[tag])
            assert len(content) == len(con_to_tag)

            for i in range(len(content)):
                txt_file.write(content[i] + ' ' + con_to_tag[i] + '\n')
            txt_file.write('\n')
    txt_file.close()


# 移动pdf文件到train和val
def move_file_to_train_and_val(from_dir, to_dir1, to_dir2):
    paths = os.listdir(from_dir)
    for i, p in enumerate(paths):
        file = os.path.join(from_dir, p)
        if i < 1700:
            shutil.copy(file, to_dir1)
        else:
            shutil.copy(file, to_dir2)
    print('finish')

def build_train_val_json_by_own_train_json(own_json_path, train_dir, val_dir, train_json_path, val_json_path):
    with open(own_json_path, 'r') as j:
        own_filename_to_info = json.load(j)
    train_filename_to_info = {}
    val_filename_to_info = {}
    train_paths = os.listdir(train_dir)
    val_paths = os.listdir(val_dir)
    for p in train_paths:
        if p.endswith('.pdf'):
            train_filename = p[:-4]
            train_filename_to_info[train_filename] = own_filename_to_info[train_filename]
    for p in val_paths:
        if p.endswith('.pdf'):
            val_filename = p[:-4]
            val_filename_to_info[val_filename] = own_filename_to_info[val_filename]
    with open(train_json_path, 'w') as j:
        json.dump(train_filename_to_info, j)
    with open(val_json_path, 'w') as j:
        json.dump(val_filename_to_info, j)
    print('finish')


def bulit_tag_json_without_space(tag_file_path, save_file_path):
    with open(tag_file_path, 'r') as j:
        name_to_info = json.load(j)
    for name in name_to_info:
        info = name_to_info[name]
        for k, v in info.items():
            if isinstance(v, list):
                for i, s in enumerate(v):
                    temp = s.split()
                    s = ''.join(temp)
                    name_to_info[name][k][i] = s
            else:
                temp = v.split()
                v = ''.join(temp)
                name_to_info[name][k] = v
    with open(save_file_path, 'w') as j:
        json.dump(name_to_info, j)

def build_perfect_word_to_tag_by_long_error(long_error_txt_path, ori_tag_file_path, save_tag_file_path):
    tag_to_cn = {'name': '姓名', 'bir': '出生年月', 'gend': '性别', 'tel': '电话', 'acad': '最高学历',
                 'nati': '籍贯', 'live': '落户市县', 'poli': '政治面貌', 'unv': '毕业院校', 'comp': '工作单位',
                 'work': '工作内容', 'post': '职务', 'proj': '项目名称', 'resp': '项目责任', 'degr': '学位',
                 'grti': '毕业时间', 'woti': '工作时间', 'prti': '项目时间'}
    with open(ori_tag_file_path, 'r') as j:
        name_to_info = json.load(j)
    with open(long_error_txt_path, 'r') as f:
        for result in f.readlines():
            line = result.split()
            try:
                if len(line) == 4:
                    file_name, ori_text, tag, text = line
                    name_to_info[file_name][tag_to_cn[tag]].remove(ori_text)
                    name_to_info[file_name][tag_to_cn[tag]].append(text)
                else:
                    print(line)
            except Exception as e:
                print(e)
    with open(save_tag_file_path, 'w') as j:
        json.dump(name_to_info, j)
    print('finish')


if __name__ == '__main__':
    start = time.time()

    pdf_dir = '/home/agwave/Data/resume/resume_train_20200121/pdf/'
    tag_file_path = '/home/agwave/Data/resume/own_all_data_perfect_0226.json'
    txt_path = '/home/agwave/Data/resume/resume_train_20200121/all_data_word_to_tag_perfect_0226.txt'
    tagging2txt_with_c(pdf_dir, tag_file_path, txt_path)

    # tag_file_path = '/home/agwave/Data/resume/own_train_data_20200207.json'
    # save_file_path = '/home/agwave/Data/resume/own_all_data_without_space.json'
    # bulit_tag_json_without_space(tag_file_path, save_file_path)

    # long_error_txt_path = 'long_text_error.txt'
    # ori_tag_file_path = '/home/agwave/Data/resume/own_all_data_without_space.json'
    # save_tag_file_path = '/home/agwave/Data/resume/own_all_data_perfect_0226.json'
    # build_perfect_word_to_tag_by_long_error(long_error_txt_path, ori_tag_file_path, save_tag_file_path)

    print(time.time() - start)
