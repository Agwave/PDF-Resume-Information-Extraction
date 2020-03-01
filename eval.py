import time
import json
import os
from tqdm import tqdm
import torch
import torch.nn as nn

from data_process import get_str_from_pdf
from model import BiLSTM_CRF
from util import get_score_by_label_pred, get_data_from_data_txt, log_sum_exp,\
    prepare_sequence, write_info_by_ix

FINAL_DATA_PATH = '/home/agwave/Data/resume/word_to_tag.txt'
SAMPLE_PDF_FILE = '/home/agwave/Data/resume/resume_train_20200121/pdf/081bfbdfaded.pdf'
SAMPLE_PDF_DIR = '/home/agwave/Data/resume/resume_train_20200121/pdf_simple/'
SAMPLE_REAL_TAG_PATH = '/home/agwave/Data/resume/resume_train_20200121/pdf_sample_real.json'

TRAIN_PDF_DIR = '/home/agwave/Data/resume/resume_train_20200121/pdf/'
TRAIN_REAL_TAG_PATH = '/home/agwave/Data/resume/resume_train_20200121/own_train_data.json'

MDOEL_PATH = 'model/model_0223.pth'
PRED_JSON_DIR = '/home/agwave/Data/resume/resume_train_20200121/pred_json/'
EMBEDDING_DIM = 100
HIDDEN_DIM = 100

torch.manual_seed(1)


def eval_one_sample():
    sample = list(get_str_from_pdf(SAMPLE_PDF_FILE))

    with open('supporting_document/word_to_ix_add_unk_0219.json') as j:
        word_to_ix = json.load(j)

    tag_to_ix = {'b-name': 0, 'i-name': 1, 'b-bir': 2, 'i-bir': 3, 'b-gend': 4, 'i-gend': 5,
                 'b-tel': 6, 'i-tel': 7, 'b-acad': 8, 'i-acad': 9, 'b-nati': 10, 'i-nati': 11,
                 'b-live': 12, 'i-live': 13, 'b-poli': 14, 'i-poli': 15, 'b-unv': 16, 'i-unv': 17,
                 'b-comp': 18, 'i-comp': 19, 'b-work': 20, 'i-work': 21, 'b-post': 22, 'i-post': 23,
                 'b-proj': 24, 'i-proj': 25, 'b-resp': 26, 'i-resp': 27, 'b-degr': 28, 'i-degr': 29,
                 'b-grti': 30, 'i-grti': 31, 'b-woti': 32, 'i-woti': 33, 'b-prti': 34, 'i-prti': 35,
                 'o': 36, '<start>': 37, '<stop>': 38, 'c-live': 39, 'c-proj': 40, 'c-woti': 41,
                 'c-post': 42, 'c-unv': 43, 'c-nati': 44, 'c-poli': 45, 'c-prti':46, 'c-comp': 47}

    ix_to_word = {}
    for k, v in tag_to_ix.items():
        ix_to_word[v] = k

    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)

    checkpoint = torch.load('model_100_all_data_0226.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    with torch.no_grad():
        precheck_sent = prepare_sequence(sample, word_to_ix)
        score, ix = model(precheck_sent)
    print(score)
    predict = []
    for i in ix:
        predict.append(ix_to_word[i])
    for i in range(len(ix)):
        print(sample[i], predict[i])

def get_score_from_model_path(model_path, tag_file, pdf_root_dir, pred_json_dir=None):
    path = os.listdir(pdf_root_dir)
    with open('supporting_document/train_word_to_tag_0223.json') as j:
        word_to_ix = json.load(j)
    tag_to_ix = {'b-name': 0, 'i-name': 1, 'b-bir': 2, 'i-bir': 3, 'b-gend': 4, 'i-gend': 5,
                 'b-tel': 6, 'i-tel': 7, 'b-acad': 8, 'i-acad': 9, 'b-nati': 10, 'i-nati': 11,
                 'b-live': 12, 'i-live': 13, 'b-poli': 14, 'i-poli': 15, 'b-unv': 16, 'i-unv': 17,
                 'b-comp': 18, 'i-comp': 19, 'b-work': 20, 'i-work': 21, 'b-post': 22, 'i-post': 23,
                 'b-proj': 24, 'i-proj': 25, 'b-resp': 26, 'i-resp': 27, 'b-degr': 28, 'i-degr': 29,
                 'b-grti': 30, 'i-grti': 31, 'b-woti': 32, 'i-woti': 33, 'b-prti': 34, 'i-prti': 35,
                 'o': 36, '<start>': 37, '<stop>': 38}
    ix_to_tag = {v: k for k, v in tag_to_ix.items()}

    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    pred_pdf_info = {}
    print('predicting...')
    for p in tqdm(path):
        if p.endswith('.pdf'):
            file_name = p[:-4]
            try:
                content = get_str_from_pdf(os.path.join(pdf_root_dir, p))
                char_list = list(content)
                with torch.no_grad():
                    precheck_sent = prepare_sequence(char_list, word_to_ix)
                    _, ix = model(precheck_sent)
                info = write_info_by_ix(ix, content, ix_to_tag)
                pred_pdf_info[file_name] = info
            except Exception as e:
                if file_name not in pred_pdf_info:
                    pred_pdf_info[file_name] = {}
                print(e)
    print('predict OK!')
    if pred_json_dir != None:
        pred_json_path = os.path.join(pred_json_dir, model_path[-4]+'.json')
        with open(pred_json_path, 'w') as j:
            json.dump(pred_pdf_info, j, ensure_ascii=False)

    with open(tag_file, 'r') as j:
        label_pdf_info = json.load(j)
    score = get_score_by_label_pred(label_pdf_info, pred_pdf_info)
    return score



if __name__ == '__main__':
    start_time = time.time()
    # score = get_score_from_model_path(MDOEL_PATH, TRAIN_REAL_TAG_PATH, TRAIN_PDF_DIR)
    # print('final score:', score)
    eval_one_sample()
    print('running time:', time.time() - start_time)