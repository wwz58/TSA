# -*- coding: utf-8 -*-
# file: td_lstm.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

# from layers.dynamic_rnn import DynamicLSTM
import pickle
import os
import torch
import torch.nn as nn


class TD_LSTM(nn.Module):
    def __init__(self, opt):
        super(TD_LSTM, self).__init__()
        with open(
                os.path.join(
                    'data',
                    f'debug{opt.debug}.{opt.dataset}.spacy.wv.{opt.embedding_dim}.pkl'
                ), 'rb') as f:
            embedding_matrix = pickle.load(f)
        self.embed = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float))
        self.lstm_l = nn.LSTM(opt.embedding_dim,
                              opt.lstm_hidden_dim,
                              num_layers=1,
                              batch_first=True)
        self.lstm_r = nn.LSTM(opt.embedding_dim,
                              opt.lstm_hidden_dim,
                              num_layers=1,
                              batch_first=True)
        self.dense = nn.Linear(opt.hidden_dim * 2, opt.num_classes)

    def forward(self, inputs):
        x_l, x_r = inputs[0], inputs[1]
        # x_l_len, x_r_len = torch.sum(x_l != 0, dim=-1), torch.sum(x_r != 0,
        #   dim=-1)
        x_l, x_r = self.embed(x_l), self.embed(x_r)
        _, (h_n_l, _) = self.lstm_l(x_l)  #, x_l_len)
        _, (h_n_r, _) = self.lstm_r(x_r)  #, x_r_len)
        h_n = torch.cat((h_n_l[0], h_n_r[0]), dim=-1)
        out = self.dense(h_n)
        return out
