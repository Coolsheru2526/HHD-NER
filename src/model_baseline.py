import torch.nn as nn
from transformers import AutoModel
from TorchCRF import CRF

class MuRIL_CRF(nn.Module):
    def __init__(self, num_tags):
        super().__init__()
        self.muril = AutoModel.from_pretrained(
            "google/muril-base-cased",
            output_hidden_states=True
        )
        self.fc = nn.Linear(768, num_tags)
        self.crf = CRF(num_tags)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.muril(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Sum last 4 hidden layers (standard practice for BERT-based models)
        x = sum(outputs.hidden_states[-4:])
        emissions = self.fc(x)

        # CRF mask: use attention_mask consistently for both train and inference
        crf_mask = attention_mask.bool()

        if labels is not None:
            # Prepare labels for CRF
            labels = labels.clone()
            labels[labels == -100] = 0   # Replace ignored indices with dummy 'O' tag
            
            # CLS token gets 'O' tag (standard practice in BERT NER)
            labels[:, 0] = 0
            crf_mask[:, 0] = True

            # Compute negative log-likelihood loss
            log_likelihood = self.crf.forward(
                emissions,
                labels,
                mask=crf_mask
            )
            loss = -log_likelihood.mean()
            return loss

        # Viterbi decoding for inference
        return self.crf.viterbi_decode(
            emissions,
            mask=crf_mask
        )

