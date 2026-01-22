import torch.nn as nn
from transformers import AutoModel


class MuRIL_Softmax(nn.Module):
    """
    MuRIL + Softmax (No CRF) for NER
    ================================
    Ablation study: Replace CRF with simple Softmax classification.
    """
    
    def __init__(self, num_tags):
        super().__init__()
        self.muril = AutoModel.from_pretrained(
            "google/muril-base-cased",
            output_hidden_states=True
        )
        self.fc = nn.Linear(768, num_tags)
        # No CRF - just use cross-entropy loss
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.muril(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Sum last 4 hidden layers (standard practice for BERT-based models)
        x = sum(outputs.hidden_states[-4:])
        logits = self.fc(x)  # (batch, seq_len, num_tags)

        if labels is not None:
            # Flatten for cross-entropy loss
            # logits: (batch * seq_len, num_tags)
            # labels: (batch * seq_len)
            loss = self.loss_fn(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
            return loss

        # Inference: argmax over tag dimension
        predictions = torch.argmax(logits, dim=-1)  # (batch, seq_len)
        
        # Convert to list of lists (matching CRF output format)
        return predictions.cpu().tolist()
