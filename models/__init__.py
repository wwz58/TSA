from torch import nn
from models.lstm import TSA_LSTM
from models.tdlstm import TD_LSTM

input_cols = {
    'lstm': ['sent_ids', 'target_ids', 'sent_mask'],
    'tdlstm': ['left_sent_ids', 'right_sent_ids']
}
model_map = {'lstm': TSA_LSTM, 'tdlstm': TD_LSTM}


def init_weight(model):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
