import os
import json
import re
from io import StringIO

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage

PDF_DIR = './tcdata/test_data/'
JSON_FILE_PATH = './test_result.json'


# PDF_DIR = '/home/agwave/Data/resume/resume_train_20200121/pdf/'
# JSON_FILE_PATH = '/home/agwave/Data/resume/resume_train_20200121/train_result3.json'


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


def gen_json(pdf_root_dir, json_file_path):
    ret = {}
    path = os.listdir(pdf_root_dir)
    for p in path:
        if p.endswith('.pdf'):
            file_name = p[:-4]
            content = pdf2strlist(os.path.join(pdf_root_dir, p))
            info = write_info(content)
            ret[file_name] = info
    json.dump(ret, open(json_file_path, 'w', encoding='utf-8'), ensure_ascii=False)

if __name__ == '__main__':
    gen_json(PDF_DIR, JSON_FILE_PATH)