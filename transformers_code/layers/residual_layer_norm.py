import torch.nn as nn


class ResidualLayerNorm(nn.Module):
    def __init__(self, d_model, dropout=0.3):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, residual):
        ## Apply the residual
        residual_applied = ?
        ln = self.layer_norm(residual_applied)
        return self.dropout(ln)
