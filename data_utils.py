import os
import pickle
import re
from collections import Counter
from xml.etree import ElementTree as ET

import numpy as np
import spacy
# from pytorch_transformers import BertTokenizer
from spacy.tokens import Doc
from torch.utils.data import Dataset


def read_ex_with_clean(fn, label2id, debug=False):
    tree = ET.parse(fn)
    root = tree.getroot()
    replace_p = re.compile(r'\xa0')
    add_blank_p = re.compile(r"([][!\"#$%&\\'()*+,\-./:;<=>?@])")
    blank_p = re.compile(r'\s+')
    for i, sentence in enumerate(root.findall('sentence')):
        if debug and i > 30:
            return
            yield
        id = sentence.get('id')
        text = sentence.find('text').text
        aspectTerms = sentence.find('aspectTerms')
        if aspectTerms is not None:
            for aspectTerm in aspectTerms.findall('aspectTerm'):
                term = aspectTerm.get('term')
                polarity = aspectTerm.get('polarity')
                if polarity == 'conflict':
                    continue
                polarity = label2id[polarity]
                start = int(aspectTerm.get('from'))
                end = int(aspectTerm.get('to'))
                left = text[:start]
                right = text[end:]
                left, term, right = [
                    replace_p.sub('', s) for s in [left, term, right]
                ]
                left, term, right = [
                    add_blank_p.sub(r' \1 ', s) for s in [left, term, right]
                ]
                left, term, right = [
                    blank_p.sub(' ', s) for s in [left, term, right]
                ]
                yield [id, left.strip(), term.strip(), right.strip(), polarity]


def pad_and_truncate(sequence,
                     maxlen,
                     dtype='int64',
                     padding='post',
                     truncating='post',
                     value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


def build_w2i_and_embedding_matrix(opt, nlp):
    if not (os.path.exists(
            os.path.join(
                'data',
                f'debug{opt.debug}.{opt.dataset}.spacy.wv.{opt.embedding_dim}.pkl'
            )) and os.path.exists(
                os.path.join('data',
                             f'debug{opt.debug}.{opt.dataset}.w2i.pkl'))):
        idx = 1
        w2i = {'[PAD]': 0}
        embedding_matrix = [np.zeros(opt.embedding_dim, dtype='float32')]
        for mode in ['train', 'test']:
            for id, left, term, right, polarity in read_ex_with_clean(
                    opt.dataset_fs[opt.dataset][mode], opt.label2id,
                    opt.debug):
                doc = nlp((left + ' ' + term + ' ' + right).strip())

                for t in doc:
                    if t not in w2i:
                        w2i[t.text] = idx
                        embedding_matrix.append(t.vector)
                        idx += 1
        w2i['[UNK]'] = idx
        embedding_matrix.append(np.zeros(opt.embedding_dim, dtype='float32'))
        embedding_matrix = np.stack(embedding_matrix)

        with open(
                os.path.join(
                    'data',
                    f'debug{opt.debug}.{opt.dataset}.spacy.wv.{opt.embedding_dim}.pkl'
                ), 'wb') as f:
            pickle.dump(embedding_matrix, f)
        with open(
                os.path.join('data',
                             f'debug{opt.debug}.{opt.dataset}.w2i.pkl'),
                'wb') as f:
            pickle.dump(w2i, f)
    else:
        with open(
                os.path.join(
                    'data',
                    f'debug{opt.debug}.{opt.dataset}.spacy.wv.{opt.embedding_dim}.pkl'
                ), 'rb') as f:
            embedding_matrix = pickle.load(f)
        with open(
                os.path.join('data',
                             f'debug{opt.debug}.{opt.dataset}.w2i.pkl'),
                'rb') as f:
            w2i = pickle.load(f)
    return embedding_matrix, w2i


class MyData(Dataset):
    def __init__(self, opt, mode='train'):
        if not os.path.exists(
                os.path.join(
                    'data', 'debug%s.%s.sent%s.target%s.%s.pkl' %
                    (opt.debug, opt.dataset, opt.sent_max_len,
                     opt.target_max_len, mode))):
            nlp = spacy.load("en_core_web_lg")
            embedding_matrix, w2i = build_w2i_and_embedding_matrix(opt, nlp)
            self.data = []

            for id, left, term, right, polarity in read_ex_with_clean(
                    opt.dataset_fs[opt.dataset][mode], opt.label2id,
                    opt.debug):
                left_toks = [token for token in nlp(left)]
                term_toks = [token for token in nlp(term)]
                right_toks = [token for token in nlp(right)]
                start = len(left_toks)
                end = start + len(term_toks)
                doc = nlp((left + ' ' + term + ' ' + right).strip())

                doc_len = len(doc)
                adj = np.eye(doc_len).astype('float32')
                for t in doc:
                    for child in t.children:
                        adj[t.i][child.i] = 1.0
                        adj[child.i][t.i] = 1.0
                padded_adj = np.zeros(
                    (opt.sent_max_len, opt.sent_max_len)).astype('float32')
                padded_adj[:doc_len, :doc_len] = adj

                left_sent_ids = pad_and_truncate(
                    [w2i[t.text] for t in left_toks + term_toks],
                    opt.left_max_len)
                right_sent_ids = pad_and_truncate(
                    [w2i[t.text] for t in term_toks + right_toks],
                    opt.right_max_len)

                sent_ids = pad_and_truncate([w2i[t.text] for t in doc],
                                            opt.sent_max_len)
                target_ids = pad_and_truncate([w2i[t.text] for t in term_toks],
                                              opt.target_max_len)
                sent_mask = pad_and_truncate([1] * doc_len,
                                             opt.sent_max_len,
                                             dtype='float32')
                target_mask = pad_and_truncate([1] * len(term_toks),
                                               opt.target_max_len,
                                               dtype='float32')
                data = {
                    "left_sent_ids": left_sent_ids,
                    "right_sent_ids": right_sent_ids,
                    "adj": padded_adj,
                    "sent_ids": sent_ids,
                    "target_ids": target_ids,
                    "sent_mask": sent_mask,
                    "target_mask": target_mask,
                    "start": start,
                    "end": end,
                    "polarity": polarity
                }
                self.data.append(data)

            with open(
                    os.path.join(
                        'data', 'debug%s.%s.sent%s.target%s.%s.pkl' %
                        (opt.debug, opt.dataset, opt.sent_max_len,
                         opt.target_max_len, mode)), 'wb') as f:
                pickle.dump(self.data, f)
        else:
            with open(
                    os.path.join(
                        'data', 'debug%s.%s.sent%s.target%s.%s.pkl' %
                        (opt.debug, opt.dataset, opt.sent_max_len,
                         opt.target_max_len, mode)), 'rb') as f:
                self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]