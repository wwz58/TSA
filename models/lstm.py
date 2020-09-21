import pickle
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from models.attention import Attention


class TSA_LSTM(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        with open(
                os.path.join(
                    'data',
                    f'debug{opt.debug}.{opt.dataset}.spacy.wv.{opt.embedding_dim}.pkl'
                ), 'rb') as f:
            emb_m = pickle.load(f)
        self.emb = nn.Embedding.from_pretrained(torch.Tensor(emb_m),
                                                freeze=False)
        self.embedding_drop = nn.Dropout(opt.embedding_drop)
        self.rnn = nn.LSTM(opt.embedding_dim,
                           opt.lstm_hidden_dim,
                           batch_first=True,
                           bidirectional=True)

        self.embedding_drop_2 = nn.Dropout(opt.embedding_drop)
        self.rnn_2 = nn.LSTM(opt.embedding_dim,
                             opt.lstm_hidden_dim,
                             batch_first=True,
                             bidirectional=True)
        self.att = Attention(2 * opt.lstm_hidden_dim, opt.embedding_drop)

        self.fc = nn.Linear(3 * 2 * opt.lstm_hidden_dim,
                            2 * opt.lstm_hidden_dim)
        self.relu = nn.ReLU()
        self.fc_drop = nn.Dropout(opt.fc_drop)
        self.classifier = nn.Linear(opt.lstm_hidden_dim * 2, opt.num_classes)

    def forward(self, inputs):
        sent, target, sent_mask = inputs
        bs = len(sent)
        sent, _ = self.rnn(self.embedding_drop(self.emb(sent)))
        target_seq, hc_n = self.rnn_2(self.embedding_drop_2(self.emb(target)))
        h_n = hc_n[0].transpose(0, 1).reshape(bs, -1)
        target = torch.cat([target_seq.max(1)[0], h_n, target_seq.mean(1)], -1)
        target = self.relu(self.fc_drop(self.fc(target)))
        rep = self.att(sent, target, sent_mask)
        logits = self.classifier(rep)
        return logits
