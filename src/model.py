import torch
import torch.nn as nn

class RCNNEmotionNet(nn.Module):
    """RCNN + Attention fusion model.

    Assumptions about acoustic feature extractor (src/features.py):
    ─ n_mfcc  = 40  MFCC coefficients
    ─ +1      =  pitch track
    ─ +n_mfcc =  spectral‑flux channels (tiled)
    Total feature dim before pooling = 40 + 1 + 40 = **81**
    After a 2×2 max‑pool, the frequency dimension halves (→ 40).
    Therefore the GRU input size = 40 * cnn_channels.
    """

    def __init__(
        self,
        n_mfcc: int = 40,
        cnn_channels: int = 64,
        rnn_hidden: int = 128,
        txt_dim: int = 384,
        n_classes: int = 6,
    ):
        super().__init__()
        # 2‑D CNN encoder
        self.conv = nn.Sequential(
            nn.Conv2d(1, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )
        # Calculate GRU input size dynamically from n_mfcc
        total_f = n_mfcc * 2 + 1        # 81
        pooled_f = total_f // 2         # 40
        gru_in = pooled_f * cnn_channels  # 40 * 64 = 2560

        self.rnn = nn.GRU(
            input_size=gru_in,
            hidden_size=rnn_hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.attn = nn.Linear(rnn_hidden * 2, 1)
        self.fc = nn.Sequential(
            nn.Linear(rnn_hidden * 2 + txt_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, spec: torch.Tensor, txt_emb: torch.Tensor):
        # spec: (B, T, F) → (B, 1, T, F)
        x = self.conv(spec.unsqueeze(1))  # (B, C, T/2, F/2)
        B, C, T, F = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T, C * F)  # (B, T, C×F)
        h, _ = self.rnn(x)                               # (B, T, 2H)
        alpha = torch.softmax(self.attn(h).squeeze(-1), dim=1)
        context = (h * alpha.unsqueeze(-1)).sum(dim=1)
        fused = torch.cat([context, txt_emb], dim=-1)
        return self.fc(fused)
