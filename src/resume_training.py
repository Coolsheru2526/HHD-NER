
import os
import json
import sys
import time
import random
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset import read_conll, encode
from model_baseline import MuRIL_CRF
from evaluation import evaluate_ner, validate_bio_sequences, compute_entity_distribution


DATA_DIR = "data/processed"
TRAIN_PATH = os.path.join(DATA_DIR, "train_split.conll")
DEV_PATH = os.path.join(DATA_DIR, "dev.conll")
SAVE_DIR = "weights"

# Resume configuration
RESUME_FROM_EPOCH = 5  
TOTAL_EPOCHS = 10     
BATCH_SIZE = 16
LR = 2e-5
PATIENCE = 2

# Tag schema
TAGS = ["O", "B-DISEASE", "I-DISEASE", "B-SYMPTOM", "I-SYMPTOM", 
        "B-CONSUMABLE", "I-CONSUMABLE", "B-PERSON", "I-PERSON"]
tag2id = {t: i for i, t in enumerate(TAGS)}
id2tag = {i: t for t, i in tag2id.items()}


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


def load_dataset(path: str, tag2id: dict):
    sentences, labels = read_conll(path)
    enc = encode(sentences, labels, tag2id)
    return NERDataset(enc["input_ids"], enc["attention_mask"], enc["labels"])


def validate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            loss = model(input_ids, attention_mask, labels)
            total_loss += loss.item()
            
            predictions = model(input_ids, attention_mask)
            
            for i in range(len(predictions)):
                pred_seq = []
                true_seq = []
                
                for j, (pred_id, true_id) in enumerate(zip(predictions[i], labels[i].cpu().tolist())):
                    if true_id != -100:
                        pred_seq.append(id2tag.get(pred_id, "O"))
                        true_seq.append(id2tag.get(true_id, "O"))
                
                all_preds.append(pred_seq)
                all_labels.append(true_seq)
    
    avg_loss = total_loss / len(dataloader)
    
    from seqeval.metrics import f1_score
    from seqeval.scheme import IOB2
    f1 = f1_score(all_labels, all_preds, mode='strict', scheme=IOB2)
    
    model.train()
    return avg_loss, f1, all_preds, all_labels


def resume_training():
    """Resume training from checkpoint."""
    
    SEED = 42
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup logging
    log_file = os.path.join(SAVE_DIR, "training_resume.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("RESUMING TRAINING FROM CHECKPOINT")
    logger.info("="*60)
    logger.info(f"Using device: {device}")
    logger.info(f"Resume from epoch: {RESUME_FROM_EPOCH}")
    logger.info(f"Target total epochs: {TOTAL_EPOCHS}")
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = load_dataset(TRAIN_PATH, tag2id)
    dev_dataset = load_dataset(DEV_PATH, tag2id)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    model = MuRIL_CRF(len(TAGS)).to(device)
    
    # Load checkpoint
    checkpoint_path = os.path.join(SAVE_DIR, f"checkpoint_epoch_{RESUME_FROM_EPOCH}.pt")
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    logger.info("[OK] Checkpoint loaded")
    
    # Load previous training history
    history_path = os.path.join(SAVE_DIR, "training_history.json")
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
        logger.info("[OK] Previous training history loaded")
        logger.info(f"Previous best F1: {history['best_f1']:.4f} at epoch {history['best_epoch']}")
        best_f1 = history['best_f1']
    else:
        logger.warning("No previous history found, starting fresh history")
        history = {
            "train_loss": [],
            "dev_loss": [],
            "dev_f1": [],
            "epoch_times": [],
            "gpu_memory_gb": [],
            "best_epoch": 0,
            "best_f1": 0.0,
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR
        }
        best_f1 = 0.0
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=1, verbose=True
    )
    
    patience_counter = 0
    
    logger.info("Resuming training...")
    logger.info("="*60)
    total_training_start = time.time()
    
    # Continue training from RESUME_FROM_EPOCH to TOTAL_EPOCHS
    for epoch in range(RESUME_FROM_EPOCH, TOTAL_EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_start_time = time.time()
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{TOTAL_EPOCHS}")
        
        for batch in progress:
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            loss = model(input_ids, attention_mask, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_train_loss = epoch_loss / len(train_loader)
        
        # Validation
        dev_loss, dev_f1, _, _ = validate(model, dev_loader, device)
        
        # Update scheduler
        scheduler.step(dev_f1)
        epoch_time = time.time() - epoch_start_time
        
        if torch.cuda.is_available():
            max_memory = torch.cuda.max_memory_allocated() / (1024**3)
            torch.cuda.reset_peak_memory_stats()
        else:
            max_memory = 0.0
        
        # Append to history
        history["train_loss"].append(avg_train_loss)
        history["dev_loss"].append(dev_loss)
        history["dev_f1"].append(dev_f1)
        history["epoch_times"].append(epoch_time)
        history["gpu_memory_gb"].append(max_memory)
        
        logger.info(f"\nEpoch {epoch+1}/{TOTAL_EPOCHS}:")
        logger.info(f"  Train Loss: {avg_train_loss:.4f}")
        logger.info(f"  Dev Loss:   {dev_loss:.4f}")
        logger.info(f"  Dev F1:     {dev_f1:.4f}")
        logger.info(f"  Time:       {epoch_time/60:.1f} min")
        logger.info(f"  GPU Memory: {max_memory:.2f} GB")
        
        # Save best model
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            history["best_epoch"] = epoch + 1
            history["best_f1"] = best_f1
            patience_counter = 0
            
            model_path = os.path.join(SAVE_DIR, "best_model.pt")
            torch.save(model.state_dict(), model_path)
            logger.info(f"  [OK] New best model saved (F1={best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(f"\nEarly stopping at epoch {epoch+1} (patience={PATIENCE})")
                break
        
        # Save checkpoint
        checkpoint_path = os.path.join(SAVE_DIR, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
    
    # Save updated history
    total_training_time = time.time() - total_training_start
    if "total_training_time_hours" in history:
        history["total_training_time_hours"] += total_training_time / 3600
    else:
        history["total_training_time_hours"] = total_training_time / 3600
    
    with open(os.path.join(SAVE_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    
    logger.info("="*60)
    logger.info("Training complete!")
    logger.info(f"Best F1: {history['best_f1']:.4f} at epoch {history['best_epoch']}")
    logger.info(f"Additional training time: {total_training_time/3600:.2f} hours")
    logger.info(f"Model saved to {SAVE_DIR}/best_model.pt")
    logger.info("="*60)
    
    return history


if __name__ == "__main__":
    resume_training()
