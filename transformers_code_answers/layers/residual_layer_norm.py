import torch.nn as nn

class ResidualLayerNorm(nn.Module):
    def __init__(self, d_model, dropout=0.3):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, residual):
        ln = self.layer_norm(x + residual)
        return self.dropout(ln)