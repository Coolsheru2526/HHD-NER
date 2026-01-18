import os
import json
import torch
from tqdm import tqdm

from dataset import read_conll, encode
from model import MuRIL_CRF


DATA_PATH = "data/processed/train.conll"
SAVE_DIR = "weights"
EPOCHS = 10
LR = 2e-5

os.makedirs(SAVE_DIR, exist_ok=True)


TAGS = ["O", "B-DISEASE", "B-SYMPTOM", "B-CONSUMABLE", "B-PERSON"]
tag2id = {t: i for i, t in enumerate(TAGS)}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


sentences, labels = read_conll(DATA_PATH)
enc = encode(sentences, labels, tag2id)

input_ids = enc["input_ids"].to(device)
attention_mask = enc["attention_mask"].to(device)
labels = enc["labels"].to(device)

print("Dataset size:", input_ids.size(0))


model = MuRIL_CRF(len(TAGS)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)


loss_history = {
    "epoch_loss": [],
    "step_loss": []
}


model.train()

for epoch in range(EPOCHS):
    epoch_loss = 0.0

    progress = tqdm(
        range(input_ids.size(0)),
        desc=f"Epoch {epoch+1}/{EPOCHS}",
        leave=True
    )

    for i in progress:
        optimizer.zero_grad()

        # single-sentence training (CRF safe)
        loss = model(
            input_ids=input_ids[i:i+1],
            attention_mask=attention_mask[i:i+1],
            labels=labels[i:i+1]
        )

        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        epoch_loss += loss_val
        loss_history["step_loss"].append(loss_val)

        progress.set_postfix(loss=f"{loss_val:.4f}")

    avg_loss = epoch_loss / input_ids.size(0)
    loss_history["epoch_loss"].append(avg_loss)

    print(f"\nEpoch {epoch+1} Average Loss: {avg_loss:.4f}")


    model_path = os.path.join(
        SAVE_DIR,
        f"muril_crf_epoch_{epoch+1}.pt"
    )
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


    with open(os.path.join(SAVE_DIR, "loss_history.json"), "w") as f:
        json.dump(loss_history, f, indent=2)

print("\nTraining completed successfully.")
