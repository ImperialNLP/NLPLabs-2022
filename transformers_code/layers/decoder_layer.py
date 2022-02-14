import torch.nn as nn
from .mha import MultiHeadAttention
from .pwffn import PWFFN
from .residual_layer_norm import ResidualLayerNorm

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.3):
        super().__init__()
        
        ## initialise 3 ResidualLayerNorm layers
        self.norm_1 = ?
        self.norm_2 = ?
        self.norm_3 = ?

        ## initialise 2 multihead attention layers
        self.mha_1 = ?
        self.mha_2 = ?
        
        ## initialise a postionwise feedforward layer
        self.ff = ?

    def forward(self, x, encoder_outputs, trg_mask, src_mask):
        # shape(x) = [B x TRG_seq_len x D]
        # shape(encoder_outputs) = [B x SRC_seq_len x D]

        ## call the first multihead attention layer (remember to pass a mask)
        masked_mha, masked_mha_attn_weights = ??
        # shape(masked_mha) = [B x TRG_seq_len x D]
        # shape(masked_mha_attn_weights) = [B x num_heads x TRG_seq_len x TRG_seq_len]

        norm1 = self.norm_1(masked_mha, x)
        # shape(norm1) = [B x TRG_seq_len x D]
        
        ## call the second multihead attention layer (remember to pass a mask)
        enc_dec_mha, enc_dec_mha_attn_weights = ??
        # shape(enc_dec_mha) = [B x TRG_seq_len x D]
        # shape(enc_dec_mha_attn_weights) = [B x num_heads x TRG_seq_len x SRC_seq_len]

        norm2 = self.norm_2(enc_dec_mha, norm1)
        # shape(norm2) = [B x TRG_seq_len x D]

        ff = self.ff(norm2)
        norm3 = self.norm_3(ff, norm2)
        # shape(ff) = [B x TRG_seq_len x D]
        # shape(norm3) = [B x TRG_seq_len x D]

        return norm3, masked_mha_attn_weights, enc_dec_mha_attn_weights