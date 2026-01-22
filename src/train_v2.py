
import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset import read_conll, encode
from model import MuRIL_CRF
from evaluation import evaluate_ner


DATA_DIR = "data/processed"
TRAIN_PATH = os.path.join(DATA_DIR, "train_split.conll")
DEV_PATH = os.path.join(DATA_DIR, "dev.conll")
SAVE_DIR = "weights"

EPOCHS = 5
BATCH_SIZE = 16
LR = 2e-5
PATIENCE = 2  # Early stopping patience

# Tag schema (must match conll_builder output)
TAGS = ["O", "B-DISEASE", "I-DISEASE", "B-SYMPTOM", "I-SYMPTOM", 
        "B-CONSUMABLE", "I-CONSUMABLE", "B-PERSON", "I-PERSON"]
tag2id = {t: i for i, t in enumerate(TAGS)}
id2tag = {i: t for t, i in tag2id.items()}

class NERDataset(Dataset):
    """PyTorch Dataset wrapper for NER data."""
    
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


def load_dataset(path: str, tag2id: dict):
    """Load and encode CoNLL file."""
    sentences, labels = read_conll(path)
    enc = encode(sentences, labels, tag2id)
    return NERDataset(enc["input_ids"], enc["attention_mask"], enc["labels"])


def validate(model, dataloader, device):
    """
    Run validation and compute F1 score.
    
    Returns:
        Tuple of (average_loss, f1_score, all_predictions, all_labels)
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Get loss
            loss = model(input_ids, attention_mask, labels)
            total_loss += loss.item()
            
            # Get predictions
            predictions = model(input_ids, attention_mask)
            
            # Convert to tag sequences (excluding -100 positions)
            for i in range(len(predictions)):
                pred_seq = []
                true_seq = []
                
                for j, (pred_id, true_id) in enumerate(zip(predictions[i], labels[i].cpu().tolist())):
                    if true_id != -100:  # Only evaluate real tokens
                        pred_seq.append(id2tag.get(pred_id, "O"))
                        true_seq.append(id2tag.get(true_id, "O"))
                
                all_preds.append(pred_seq)
                all_labels.append(true_seq)
    
    avg_loss = total_loss / len(dataloader)
    
    # Compute F1 using seqeval
    from seqeval.metrics import f1_score
    from seqeval.scheme import IOB2
    f1 = f1_score(all_labels, all_preds, mode='strict', scheme=IOB2)
    
    model.train()
    return avg_loss, f1, all_preds, all_labels

def train():
    """Main training function with batching and validation."""
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_v2] Using device: {device}")
    
    # Load datasets
    print("[train_v2] Loading datasets...")
    train_dataset = load_dataset(TRAIN_PATH, tag2id)
    dev_dataset = load_dataset(DEV_PATH, tag2id)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"[train_v2] Train: {len(train_dataset)} sentences, Dev: {len(dev_dataset)} sentences")
    
    # Initialize model
    model = MuRIL_CRF(len(TAGS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=1, verbose=True
    )
    
    # Training history
    history = {
        "train_loss": [],
        "dev_loss": [],
        "dev_f1": [],
        "best_epoch": 0,
        "best_f1": 0.0
    }
    
    best_f1 = 0.0
    patience_counter = 0
    
    print("[train_v2] Starting training...")
    print("="*60)
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in progress:
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            loss = model(input_ids, attention_mask, labels)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_train_loss = epoch_loss / len(train_loader)
        
        # Validation
        dev_loss, dev_f1, _, _ = validate(model, dev_loader, device)
        
        # Update scheduler
        scheduler.step(dev_f1)
        
        history["train_loss"].append(avg_train_loss)
        history["dev_loss"].append(dev_loss)
        history["dev_f1"].append(dev_f1)
        
        print(f"\nEpoch {epoch+1}: Train Loss={avg_train_loss:.4f}, "
              f"Dev Loss={dev_loss:.4f}, Dev F1={dev_f1:.4f}")
        
        # Save best model
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            history["best_epoch"] = epoch + 1
            history["best_f1"] = best_f1
            patience_counter = 0
            
            model_path = os.path.join(SAVE_DIR, "best_model.pt")
            torch.save(model.state_dict(), model_path)
            print(f"  -> New best model saved (F1={best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n[train_v2] Early stopping at epoch {epoch+1}")
                break
        
        # Save checkpoint
        checkpoint_path = os.path.join(SAVE_DIR, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
    
    # Save training history
    with open(os.path.join(SAVE_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    
    print("="*60)
    print(f"[train_v2] Training complete!")
    print(f"[train_v2] Best F1: {history['best_f1']:.4f} at epoch {history['best_epoch']}")
    print(f"[train_v2] Model saved to {SAVE_DIR}/best_model.pt")
    
    return history


if __name__ == "__main__":
    train()
