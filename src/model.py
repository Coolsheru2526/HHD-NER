import torch.nn as nn
from transformers import AutoModel
from torchcrf import CRF

class MuRIL_CRF(nn.Module):
    def __init__(self, num_tags):
        super().__init__()
        self.muril = AutoModel.from_pretrained(
            "google/muril-base-cased",
            output_hidden_states=True
        )
        self.fc = nn.Linear(768, num_tags)
        self.crf = CRF(num_tags, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.muril(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        x = sum(outputs.hidden_states[-4:])
        emissions = self.fc(x)

        if labels is not None:
            # Build CRF mask from labels
            crf_mask = labels != -100

            crf_mask[:, 0] = True

            labels = labels.clone()
            labels[labels == -100] = 0   # dummy 'O' tag
            labels[:, 0] = 0             # CLS gets 'O'

            return -self.crf(
                emissions,
                labels,
                mask=crf_mask
            )

        return self.crf.decode(
            emissions,
            mask=attention_mask.bool()
        )

