import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset import read_conll, encode
from model import MuRIL_NER


DATA_DIR = "data/processed"
TRAIN_PATH = os.path.join(DATA_DIR, "train_split.conll")
DEV_PATH = os.path.join(DATA_DIR, "dev.conll")
SAVE_DIR = "weights_v1"

EPOCHS = 25
BATCH_SIZE = 16
LR = 2e-5

# ---------------- TAGS ----------------
TAGS = [
    "O",
    "B-DISEASE", "I-DISEASE",
    "B-SYMPTOM", "I-SYMPTOM",
    "B-CONSUMABLE", "I-CONSUMABLE",
    "B-PERSON", "I-PERSON"
]

tag2id = {t: i for i, t in enumerate(TAGS)}
id2tag = {i: t for t, i in tag2id.items()}


# ---------------- DATASET ----------------
class NERDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx]
        }


def load_dataset(path, tag2id):
    sentences, labels = read_conll(path)
    enc = encode(sentences, labels, tag2id)
    return NERDataset(
        enc["input_ids"],
        enc["attention_mask"],
        enc["labels"]
    )


# ---------------- VALIDATION ----------------
def validate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            loss = model(input_ids, attention_mask, labels)
            total_loss += loss.item()

            predictions = model(input_ids, attention_mask)

            for i in range(len(predictions)):
                pred_seq, true_seq = [], []

                for pred_id, true_id in zip(
                    predictions[i],
                    labels[i].cpu().tolist()
                ):
                    if true_id != -100:
                        pred_seq.append(id2tag[pred_id])
                        true_seq.append(id2tag[true_id])

                all_preds.append(pred_seq)
                all_labels.append(true_seq)

    avg_loss = total_loss / len(dataloader)

    from seqeval.metrics import f1_score
    from seqeval.scheme import IOB2
    f1 = f1_score(all_labels, all_preds, mode="strict", scheme=IOB2)

    model.train()
    return avg_loss, f1


# ---------------- TRAIN ----------------
def train():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Using device: {device}")

    print("[train] Loading data...")
    train_dataset = load_dataset(TRAIN_PATH, tag2id)
    dev_dataset = load_dataset(DEV_PATH, tag2id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print(f"[train] Train: {len(train_dataset)} | Dev: {len(dev_dataset)}")

    model = MuRIL_NER(len(TAGS)).to(device)

    optimizer = torch.optim.AdamW([
        {"params": model.muril.parameters(), "lr": 2e-5},
        {"params": model.bilstm.parameters(), "lr": 1e-3},
        {"params": model.attention.parameters(), "lr": 1e-3},
        {"params": model.cnns.parameters(), "lr": 1e-3},
        {"params": model.fc.parameters(), "lr": 1e-3},
        {"params": model.crf.parameters(), "lr": 5e-4},
    ])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1, verbose=True
    )

    history = {
        "train_loss": [],
        "dev_loss": [],
        "dev_f1": [],
        "best_f1": 0.0,
        "best_epoch": 0
    }

    best_f1 = 0.0

    print("[train] Starting training")
    print("=" * 60)

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in loop:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            loss = model(input_ids, attention_mask, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = epoch_loss / len(train_loader)
        dev_loss, dev_f1 = validate(model, dev_loader, device)

        scheduler.step(dev_f1)

        history["train_loss"].append(avg_train_loss)
        history["dev_loss"].append(dev_loss)
        history["dev_f1"].append(dev_f1)

        print(
            f"\nEpoch {epoch+1}: "
            f"Train Loss={avg_train_loss:.4f} | "
            f"Dev Loss={dev_loss:.4f} | "
            f"Dev F1={dev_f1:.4f}"
        )

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            history["best_f1"] = best_f1
            history["best_epoch"] = epoch + 1

            torch.save(
                model.state_dict(),
                os.path.join(SAVE_DIR, "best_model.pt")
            )
            print(f" Saved new best model (F1={best_f1:.4f})")

        torch.save(
            model.state_dict(),
            os.path.join(SAVE_DIR, f"checkpoint_epoch_{epoch+1}.pt")
        )

    with open(os.path.join(SAVE_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print("=" * 60)
    print(f"[train] Finished. Best F1={history['best_f1']:.4f} "
          f"@ epoch {history['best_epoch']}")


if __name__ == "__main__":
    train()
