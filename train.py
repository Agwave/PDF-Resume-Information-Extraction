import time
import os
import json
import random

import torch
import torch.nn as nn
import torch.optim as optim

from util import get_data_from_data_txt, get_word_to_ix, \
    prepare_sequence, get_score_by_label_pred, write_info_by_ix_plus
from data_process import get_str_from_pdf
from model import BiLSTM_CRF
from tqdm import tqdm


DATA_WITH_C_PATH = '/home/agwave/Data/resume/resume_train_20200121/all_data_word_to_tag_c_0226.txt'
DATA_PERFECT_PATH = '/home/agwave/Data/resume/resume_train_20200121/all_data_word_to_tag_perfect_0226.txt'
FINAL_DATA_PATH = '/home/agwave/Data/resume/word_to_tag.txt'
TRAIN_WORD_TO_TAG_PATH = '/home/agwave/Data/resume/train_word_to_tag_0222.txt'
TRAIN_JSON_PATH = '/home/agwave/Data/resume/resume_train_20200121/own_train_data.json'
TRAIN_PDF_DIR = '/home/agwave/Data/resume/resume_train_20200121/pdf/'

def get_score_by_model(model, train_json_path, pdf_root_dir):
    pdf_path = os.listdir(pdf_root_dir)
    random.shuffle(pdf_path)
    path = pdf_path[:300]
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
    ix_to_tag = {v: k for k, v in tag_to_ix.items()}

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
                info = write_info_by_ix_plus(ix, content, ix_to_tag)
                pred_pdf_info[file_name] = info
            except Exception as e:
                if file_name not in pred_pdf_info:
                    pred_pdf_info[file_name] = {}
                print(e)
    print('predict OK!')

    with open(train_json_path, 'r') as j:
        label_pdf_info = json.load(j)
    score = get_score_by_label_pred(label_pdf_info, pred_pdf_info)
    return score


def train_all_data():
    embedding_dim = 100
    hidden_dim = 100
    max_score = 0
    unimprove_time = 0
    model_1_epoch = 'model/model_2_epoch_0301.pth'
    model_save_path = 'model/model_100_all_data_0301.pth'
    # Make up some training data
    training_data = get_data_from_data_txt(DATA_PERFECT_PATH)

    word_to_ix = get_word_to_ix(training_data, min_word_freq=1)

    tag_to_ix = {'b-name': 0, 'i-name': 1, 'b-bir': 2, 'i-bir': 3, 'b-gend': 4, 'i-gend': 5,
                 'b-tel': 6, 'i-tel': 7, 'b-acad': 8, 'i-acad': 9, 'b-nati': 10, 'i-nati': 11,
                 'b-live': 12, 'i-live': 13, 'b-poli': 14, 'i-poli': 15, 'b-unv': 16, 'i-unv': 17,
                 'b-comp': 18, 'i-comp': 19, 'b-work': 20, 'i-work': 21, 'b-post': 22, 'i-post': 23,
                 'b-proj': 24, 'i-proj': 25, 'b-resp': 26, 'i-resp': 27, 'b-degr': 28, 'i-degr': 29,
                 'b-grti': 30, 'i-grti': 31, 'b-woti': 32, 'i-woti': 33, 'b-prti': 34, 'i-prti': 35,
                 'o': 36, '<start>': 37, '<stop>': 38, 'c-live': 39, 'c-proj': 40, 'c-woti': 41,
                 'c-post': 42, 'c-unv': 43, 'c-nati': 44, 'c-poli': 45, 'c-prti':46, 'c-comp': 47}

    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, embedding_dim, hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Make sure prepare_sequence from earlier in the LSTM section is loaded
    for epoch in range(
            30):  # again, normally you would NOT do 300 epochs, it is toy data
        print("---------------------")
        print("running epon : ", epoch + 1)
        start_time = time.time()
        for sentence, tags in tqdm(training_data):
            model.zero_grad()
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
            loss = model.neg_log_likelihood(sentence_in, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 15)
            optimizer.step()

        cur_epoch_score = get_score_by_model(model, TRAIN_JSON_PATH, TRAIN_PDF_DIR)
        print('score', cur_epoch_score)
        print('running time:', time.time() - start_time)
        print()
        if epoch == 1:
            torch.save({
                'model_state_dict': model.state_dict()
            }, model_1_epoch)
        if cur_epoch_score > max_score:
            unimprove_time = 0
            max_score = cur_epoch_score
            torch.save({
                'model_state_dict': model.state_dict(),
            }, model_save_path)
        else:
            unimprove_time += 1
            if unimprove_time > 1:
                print('score down, break!')
                break

def train_and_val():
    embedding_dim = 100
    hidden_dim = 100
    model_load_path = None
    best_model_save_path = 'model/model_100_best_0223.pth'
    max_score = 0
    stop_epoch = 30
    unimprove_time = 0
    val_json_path = '/home/agwave/Data/resume/val_0222.json'
    val_pdf_dir = '/home/agwave/Data/resume/val_0222/'

    training_data = get_data_from_data_txt(TRAIN_WORD_TO_TAG_PATH)
    with open('supporting_document/train_word_to_tag_0223.json', 'r') as j:
        word_to_ix = json.load(j)
    tag_to_ix = {'b-name': 0, 'i-name': 1, 'b-bir': 2, 'i-bir': 3, 'b-gend': 4, 'i-gend': 5,
                 'b-tel': 6, 'i-tel': 7, 'b-acad': 8, 'i-acad': 9, 'b-nati': 10, 'i-nati': 11,
                 'b-live': 12, 'i-live': 13, 'b-poli': 14, 'i-poli': 15, 'b-unv': 16, 'i-unv': 17,
                 'b-comp': 18, 'i-comp': 19, 'b-work': 20, 'i-work': 21, 'b-post': 22, 'i-post': 23,
                 'b-proj': 24, 'i-proj': 25, 'b-resp': 26, 'i-resp': 27, 'b-degr': 28, 'i-degr': 29,
                 'b-grti': 30, 'i-grti': 31, 'b-woti': 32, 'i-woti': 33, 'b-prti': 34, 'i-prti': 35,
                 'o': 36, '<start>': 37, '<stop>': 38}
    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, embedding_dim, hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    start_epoch = 0
    if model_load_path != None:
        print('load model...')
        checkpoint = torch.load(model_load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    preliminary_score = get_score_by_model(model, val_json_path, val_pdf_dir)
    print('preliminary score:', preliminary_score)

    for epoch in range(start_epoch, stop_epoch):
        print("---------------------")
        print("running epoch : ", epoch)
        start_time = time.time()
        for sentence, tags in tqdm(training_data):
            model.zero_grad()
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
            loss = model.neg_log_likelihood(sentence_in, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        cur_epoch_score = get_score_by_model(model, val_json_path, val_pdf_dir)
        print('score', cur_epoch_score)
        print('running time:', time.time() - start_time)
        if cur_epoch_score > max_score:
            unimprove_time = 0
            max_score = cur_epoch_score
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch
            }, best_model_save_path)
            print('save best model successfully.')
        else:
            break

if __name__ == '__main__':
    train_all_data()