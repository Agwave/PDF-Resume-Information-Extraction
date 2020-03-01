import torch
import torch.nn as nn

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage


import os
import json
from io import StringIO



# PDF_DIR = './tcdata/test_data/'
# JSON_FILE_PATH = './test_result.json'

PDF_DIR = '/home/agwave/Data/resume/resume_train_20200121/pdf_simple/'
JSON_FILE_PATH = '/home/agwave/Data/resume/resume_train_20200121/pdf_simple_file.json'

EMBEDDING_DIM = 100
HIDDEN_DIM = 100
MODEL_PARM_PATH = 'model_2_epoch_0301.pth'


def argmax(vec):
    _, idx = torch.max(vec, 1)
    return idx.item()

def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

def prepare_sequence(seq, to_ix):
    idxs = [to_ix.get(w, to_ix['<unk>']) for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix['<start>'], :] = -10000
        self.transitions.data[:, tag_to_ix['<stop>']] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # '<start>' has all of the score.
        init_alphas[0][self.tag_to_ix['<start>']] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            emit_score = feat.view(-1, 1)
            tag_var = forward_var + self.transitions + emit_score
            max_tag_var, _ = torch.max(tag_var, dim=1)
            tag_var = tag_var - max_tag_var.view(-1, 1)
            forward_var = max_tag_var + torch.log(torch.sum(torch.exp(tag_var), dim=1)).view(1, -1)

        terminal_var = forward_var + self.transitions[self.tag_to_ix['<stop>']].view(1, -1)
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix['<start>']], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix['<stop>'], tags[-1]]
        return score

    def _viterbi_decode(self, feats):

        backpointers = []
        # analogous to forward
        init_vvars = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        init_vvars[0][self.tag_to_ix['<start>']] = 0
        forward_var = init_vvars
        for feat in feats:
            next_tag_var = forward_var.view(1, -1).expand(self.tagset_size, self.tagset_size) + self.transitions
            _, bptrs_t = torch.max(next_tag_var, dim=1)
            bptrs_t = bptrs_t.squeeze().data.cpu().numpy()
            next_tag_var = next_tag_var.data.cpu().numpy()
            viterbivars_t = next_tag_var[range(len(bptrs_t)), bptrs_t]
            viterbivars_t = torch.FloatTensor(viterbivars_t)
            forward_var = viterbivars_t + feat
            backpointers.append(bptrs_t)

        terminal_var = forward_var + self.transitions[self.tag_to_ix['<stop>']]
        terminal_var.data[self.tag_to_ix['<stop>']] = -10000.
        terminal_var.data[self.tag_to_ix['<start>']] = -10000.
        best_tag_id = argmax(terminal_var.unsqueeze(0))
        path_score = terminal_var[best_tag_id]
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        assert start == self.tag_to_ix['<start>']
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


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
    return ret

def write_info_by_ix(ix, content, ix_to_tag):
    tags = [ix_to_tag[i] for i in ix]
    fine_new = True
    start_idxs, end_idxs, pred_tags = [], [], []
    for i, tag in enumerate(tags):
        if fine_new:
            if tag[0] == 'b':
                start_idxs.append(i)
                fine_new = False
                pred_tags.append(tag[2:])
        else:
            if tag != 'i-' + pred_tags[-1]:
                end_idxs.append(i)
                if tag[0] == 'b':
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


def gen_json(pdf_root_dir, json_file_path):
    ret = {}
    path = os.listdir(pdf_root_dir)
    word_to_ix_path = 'word_to_ix_add_unk_0219.json'

    with open(word_to_ix_path) as j:
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

    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
    checkpoint = torch.load(MODEL_PARM_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])

    for p in path:
        if p.endswith('.pdf'):
            file_name = p[:-4]
            try:
                content = get_str_from_pdf(os.path.join(pdf_root_dir, p))
                char_list = list(content)
                with torch.no_grad():
                    precheck_sent = prepare_sequence(char_list, word_to_ix)
                    score, ix = model(precheck_sent)
                info = write_info_by_ix_plus(ix, content, ix_to_tag)
                ret[file_name] = info
            except Exception as e:
                if file_name not in ret:
                    ret[file_name] = {}
                print(e)
    json.dump(ret, open(json_file_path, 'w', encoding='utf-8'), ensure_ascii=False)

if __name__ == '__main__':
    gen_json(PDF_DIR, JSON_FILE_PATH)