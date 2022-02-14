# %%
import torch
import torch.nn as nn
import math as m
import torch.nn.functional as F

# %%


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=4, num_heads=2, dropout=0.3):
        super().__init__()

        # d_q, d_k, d_v
        self.d = d_model // num_heads

        self.d_model = d_model
        self.num_heads = num_heads

        self.dropout = nn.Dropout(dropout)

        self.linear_Qs = nn.Linear(d_model, d_model)
        self.linear_Ks = nn.Linear(d_model, d_model)
        self.linear_Vs = nn.Linear(d_model, d_model)

        self.mha_linear = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # shape(Q) = [B x seq_len x D/num_heads]
        # shape(K, V) = [B x seq_len x D/num_heads]
        # shape(Q, K, V) = [B x num_heads x seq_len x D/num_heads]

        Q_K_matmul = torch.matmul(Q, K.permute(0, 1, 3, 2))
        scores = Q_K_matmul / m.sqrt(self.d)
        # shape(scores) = [B x num_heads x seq_len x seq_len]

        if mask is not None:
            scores = scores.masked_fill(mask == False, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        # shape(attention_weights) = [B x num_heads x seq_len x seq_len]

        output = torch.matmul(attention_weights, V)
        # shape(output) = [B x num_heads x seq_len x D/num_heads]

        return output, attention_weights

    def forward(self, pre_q, pre_k, pre_v, mask=None):
        # shape of pre_q, pre_k, pre_v depends on when we're using this module.
        # If we're in the encoder:              shape(pre_q) = shape(pre_k) = shape(pre_v) = [B x SRC_seq_len x D]
        # If we're in the decoder MHA:          shape(pre_q) = shape(pre_k) = shape(pre_v) = [B x TRG_seq_len x D]
        # If we're in the decoder Cross MHA:    shape(pre_q) = [B x TRG_seq_len x D]. shape(pre_k) = shape(pre_v) = [B x SRC_seq_len x D]

        Q = self.linear_Qs(pre_q)
        K = self.linear_Qs(pre_k)
        V = self.linear_Qs(pre_v)
        # shape(Q, K, V) = [B x seq_len x D]

        batch_size = Q.shape[0]

        Q = torch.reshape(Q, (batch_size, self.num_heads, -1, self.d))
        K = torch.reshape(K, (batch_size, self.num_heads, -1, self.d))
        V = torch.reshape(V, (batch_size, self.num_heads, -1, self.d))
        # shape(Q, K, V) = [B x num_heads x seq_len x D/num_heads]

        output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask=mask)
        # shape(output) = [B x num_heads x seq_len x D/num_heads]
        # shape(attn_weights) = [B x num_heads x seq_len x seq_len]

        output = torch.reshape(output, (batch_size, -1, self.d_model))
        projection = self.dropout(self.mha_linear(output))
        # shape(output) = [B x seq_len x D]

        return projection, attn_weights
