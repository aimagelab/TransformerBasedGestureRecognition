import torch
import torch.nn as nn
import numpy as np

def position_embedding(input, d_model):
    input = input.view(-1, 1)
    dim = torch.arange(d_model // 2, dtype=torch.float32, device=input.device).view(1, -1)
    sin = torch.sin(input / 10000 ** (2 * dim / d_model))
    cos = torch.cos(input / 10000 ** (2 * dim / d_model))

    out = torch.zeros((input.shape[0], d_model), device=input.device)
    out[:, ::2] = sin
    out[:, 1::2] = cos
    return out

def sinusoid_encoding_table(max_len, d_model):
    pos = torch.arange(max_len, dtype=torch.float32)
    out = position_embedding(pos, d_model)
    return out

class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention
    """

    def __init__(self, d_model, d_k, d_v, h):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        """
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights(gain=1.0)

    def init_weights(self, gain=1.0):
        nn.init.xavier_normal_(self.fc_q.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_k.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_v.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_o.weight, gain=gain)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values):
        """
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :return:
        """
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)

        att = torch.softmax(att, -1)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention layer with Dropout and Layer Normalization.
    """

    def __init__(self, d_model, d_k, d_v, h, dff=2048, dropout=.1):
        super(MultiHeadAttention, self).__init__()

        self.attention = ScaledDotProductAttention(d_model=d_model, d_k=d_k, d_v=d_v, h=h)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        # self.layer_norm2 = nn.LayerNorm(d_model)

        self.fc = nn.Sequential(*[nn.Linear(d_model, dff), nn.ReLU(inplace=True), nn.Dropout(p=dropout),
                                  nn.Linear(dff, d_model)])

    def forward(self, queries, keys, values):
        att = self.attention(queries, keys, values)
        att = self.dropout(att)
        # att = self.layer_norm(queries + att)
        att = self.fc(att)
        att = self.dropout(att)
        return self.layer_norm(queries + att)

class EncoderSelfAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_head, dff=2048, dropout_transformer=.1, n_module=6):
        super(EncoderSelfAttention, self).__init__()
        self.encoder = nn.ModuleList([MultiHeadAttention(d_model, d_k, d_v, n_head, dff, dropout_transformer)
                                      for _ in range(n_module)])
    def forward(self, x):
        in_encoder = x + sinusoid_encoding_table(x.shape[1], x.shape[2]).expand(x.shape).cuda()
        for l in self.encoder:
            in_encoder = l(in_encoder, in_encoder, in_encoder)
        return in_encoder
