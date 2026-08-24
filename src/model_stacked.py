import torch
import torch.nn as nn
from transformers import AutoModel
from TorchCRF import CRF


class SelfAttention(nn.Module):
    """Token-level self-attention mechanism."""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x, mask):
        # (Batch, Seq, Hidden)
        h = torch.tanh(self.proj(x))
        # (Batch, Seq, 1) -> (Batch, Seq)
        scores = self.score(h).squeeze(-1)
        # Apply mask
        scores = scores.masked_fill(~mask, -1e9)
        # Softmax weights
        weights = torch.softmax(scores, dim=1)
        # Apply weights to values
        return x * weights.unsqueeze(-1)


class MuRIL_NER(nn.Module):
    """MuRIL-based NER model with BiLSTM, Multi-scale CNN, Attention, and CRF."""
    
    def __init__(
        self,
        num_tags,
        lstm_hidden=256,
        cnn_filters=128,
        kernel_sizes=(3, 5)
    ):
        super().__init__()

        # MuRIL encoder
        self.muril = AutoModel.from_pretrained(
            "google/muril-base-cased",
            output_hidden_states=True
        )

        muril_dim = 768

        # BiLSTM for sequential modeling
        self.bilstm = nn.LSTM(
            muril_dim,
            lstm_hidden,
            batch_first=True,
            bidirectional=True
        )

        lstm_out_dim = lstm_hidden * 2

        # Attention layer to focus on key entities
        self.attention = SelfAttention(lstm_out_dim)

        # Multi-scale CNN to capture local dependencies (different n-gram sizes)
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

        # Final linear mapping and CRF
        self.fc = nn.Linear(cnn_out_dim, num_tags)
        self.crf = CRF(num_tags)

    def forward(self, input_ids, attention_mask, labels=None):
        # 1. MuRIL Feature Extraction
        outputs = self.muril(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Using sum of last 4 layers for richer representation
        x = sum(outputs.hidden_states[-4:])  # (B, T, 768)

        # 2. Sequential Modeling
        x, _ = self.bilstm(x)  # (B, T, 512)

        # 3. Attention Focus
        x = self.attention(x, attention_mask.bool())

        # 4. Multi-scale CNN Feature Extraction
        # Transpose for Conv1d: (Batch, Hidden, SeqLen)
        x = x.transpose(1, 2)
        cnn_features = [torch.relu(conv(x)) for conv in self.cnns]
        # Concatenate multi-scale features: (Batch, FilterSum, SeqLen)
        x = torch.cat(cnn_features, dim=1)
        # Transpose back: (Batch, SeqLen, FilterSum)
        x = x.transpose(1, 2)

        # 5. Emission layer
        emissions = self.fc(x)

        # 6. CRF Mask Calibration
        # Ensure first token ([CLS]) is masked for decoding, but allowed for loss?
        # Standard practice: mask pads only.
        crf_mask = attention_mask.bool()
        crf_mask[:, 0] = True # Always include CLS as valid sequence start if needed

        if labels is not None:
            # Training mode: Loss calculation
            # Align labels: mapping -100 to 0 (ignored by mask)
            labels = labels.clone()
            labels[labels == -100] = 0
            labels[:, 0] = 0 # CLS is usually 'O'

            loss = -self.crf(
                emissions,
                labels,
                mask=crf_mask
            )
            return loss.mean()

        # Inference mode: Viterbi decoding
        return self.crf.viterbi_decode(
            emissions,
            mask=crf_mask
        )
