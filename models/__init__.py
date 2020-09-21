from torch import nn
from models.lstm import TSA_LSTM
input_cols = {'lstm': ['sent_ids', 'target_ids', 'sent_mask']}


def init_weight(model):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
