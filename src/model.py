import torch
import torch.nn as nn
from transformers import AutoModel
from torchcrf import CRF


# ---------------- Self-Attention ----------------
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x, mask):
        """
        x: (B, T, H)
        mask: (B, T)
        """
        h = torch.tanh(self.proj(x))
        scores = self.score(h).squeeze(-1)

        scores = scores.masked_fill(~mask, -1e9)
        attn = torch.softmax(scores, dim=1)

        return x * attn.unsqueeze(-1)


# ---------------- Main Model ----------------
class MuRIL_NER(nn.Module):
    def __init__(
        self,
        num_tags,
        lstm_hidden=256,
        cnn_filters=128,
        kernel_sizes=(3, 5)
    ):
        super().__init__()

        # MuRIL
        self.muril = AutoModel.from_pretrained(
            "google/muril-base-cased",
            output_hidden_states=True
        )

        muril_dim = 768

        # BiLSTM
        self.bilstm = nn.LSTM(
            muril_dim,
            lstm_hidden,
            batch_first=True,
            bidirectional=True
        )

        lstm_out_dim = lstm_hidden * 2

        # Attention
        self.attention = SelfAttention(lstm_out_dim)

        # CNN over sequence
        self.cnns = nn.ModuleList([
            nn.Conv1d(
                in_channels=lstm_out_dim,
                out_channels=cnn_filters,
                kernel_size=k,
                padding=k // 2
            )
            for k in kernel_sizes
        ])

        cnn_out_dim = cnn_filters * len(kernel_sizes)

        # Fully connected
        self.fc = nn.Linear(cnn_out_dim, num_tags)

        # CRF
        self.crf = CRF(num_tags, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):

        # -------- MuRIL --------
        outputs = self.muril(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # 🔑 Paper trick: sum last 4 layers
        x = sum(outputs.hidden_states[-4:])  # (B, T, 768)

        # -------- BiLSTM --------
        x, _ = self.bilstm(x)  # (B, T, 2H)

        # -------- Attention --------
        x = self.attention(x, attention_mask.bool())

        # -------- CNN --------
        x = x.transpose(1, 2)  # (B, C, T)

        cnn_feats = []
        for conv in self.cnns:
            f = torch.relu(conv(x))
            cnn_feats.append(f)

        x = torch.cat(cnn_feats, dim=1)
        x = x.transpose(1, 2)  # (B, T, C')

        # -------- Emissions --------
        emissions = self.fc(x)

        # -------- CRF --------
        if labels is not None:
            # CRF mask
            crf_mask = labels != -100
            crf_mask[:, 0] = True  # CRF constraint

            labels = labels.clone()
            labels[labels == -100] = 0  # dummy tag
            labels[:, 0] = 0

            return -self.crf(
                emissions,
                labels,
                mask=crf_mask
            )

        crf_mask = attention_mask.bool()
        crf_mask[:, 0] = True
        return self.crf.decode(emissions, mask=crf_mask)