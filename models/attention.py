import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hidden_dim, att_drop):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.q_w, self.k_w, self.v_w = [
            nn.Linear(hidden_dim, hidden_dim) for _ in range(3)
        ]
        self.drop = nn.Dropout(att_drop)

    def forward(self, sent, target, sent_mask):
        q = self.q_w(target).unsqueeze(1)  # B 1 C
        k = self.k_w(sent)  # BLC
        v = self.v_w(sent)
        score_mask = (1.0 - sent_mask) * -10000.0
        # B 1 L
        score = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.hidden_dim)
        probs = self.drop(F.softmax(score, -1))
        # B 1 C
        out = torch.matmul(probs, v).squeeze(1)
        return out
