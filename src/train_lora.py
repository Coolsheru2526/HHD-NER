
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
from model_lora import MuRIL_LoRA_CRF  # Use LoRA model
from evaluation import evaluate_ner, validate_bio_sequences, compute_entity_distribution


DATA_DIR = "data/processed"
TRAIN_PATH = os.path.join(DATA_DIR, "train_split.conll")
DEV_PATH = os.path.join(DATA_DIR, "dev.conll")
SAVE_DIR = "weights/lora"  # Different save directory

# LoRA-specific hyperparameters
EPOCHS = 5  # Same as baseline for fair comparison
BATCH_SIZE = 16
LR = 2e-4  # Typical LoRA learning rate (safer than 3e-4)
PATIENCE = 2

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


def count_parameters(model):
    """Count trainable and total parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_pct": 100 * trainable / total if total > 0 else 0
    }


def train():
    """Main training function with LoRA."""
    
    SEED = 42
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    log_file = os.path.join(SAVE_DIR, "training.log")
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
    logger.info("LoRA FINE-TUNING EXPERIMENT")
    logger.info("="*60)
    logger.info(f"Using device: {device}")
    logger.info(f"Random seed: {SEED}")
    logger.info(f"Training log: {log_file}")
    
    logger.info("Validating training data...")
    _, train_labels_check = read_conll(TRAIN_PATH)
    validation = validate_bio_sequences(train_labels_check)
    
    if not validation["is_valid"]:
        logger.error(f"Found {len(validation['violations'])} BIO violations in training data!")
        logger.error("Please fix data before training. Exiting.")
        sys.exit(1)
    
    logger.info("[OK] BIO sequences validated")
    
    # Check entity distribution
    dist = compute_entity_distribution(train_labels_check)
    logger.info("Entity distribution in training set:")
    for entity, count in sorted(dist.items(), key=lambda x: -x[1]):
        logger.info(f"  {entity}: {count}")
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = load_dataset(TRAIN_PATH, tag2id)
    dev_dataset = load_dataset(DEV_PATH, tag2id)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    logger.info(f"Train: {len(train_dataset)} sentences, Dev: {len(dev_dataset)} sentences")
    logger.info(f"Batch size: {BATCH_SIZE}, Learning rate: {LR}, Epochs: {EPOCHS}")
    
    # Initialize model with LoRA
    logger.info("Initializing model with LoRA adapters...")
    model = MuRIL_LoRA_CRF(len(TAGS)).to(device)
    logger.info("[OK] LoRA adapters applied to MuRIL")
    
    # Count parameters
    param_stats = count_parameters(model)
    logger.info("Model parameters:")
    logger.info(f"  Total:     {param_stats['total']:,}")
    logger.info(f"  Trainable: {param_stats['trainable']:,} ({param_stats['trainable_pct']:.2f}%)")
    logger.info(f"  Frozen:    {param_stats['frozen']:,}")
    
    # Optimizer - only trainable parameters
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )
    
    history = {
        "experiment": "lora",
        "train_loss": [],
        "dev_loss": [],
        "dev_f1": [],
        "epoch_times": [],
        "gpu_memory_gb": [],
        "best_epoch": 0,
        "best_f1": 0.0,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "use_lora": True,
        "lora_rank": 8,
        "lora_alpha": 16,
        "trainable_params": param_stats['trainable'],
        "total_params": param_stats['total']
    }
    
    best_f1 = 0.0
    patience_counter = 0
    
    logger.info("Starting training...")
    logger.info("="*60)
    total_training_start = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_start_time = time.time()
        
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
        epoch_time = time.time() - epoch_start_time
        
        if torch.cuda.is_available():
            max_memory = torch.cuda.max_memory_allocated() / (1024**3)  # Convert to GB
            torch.cuda.reset_peak_memory_stats()
        else:
            max_memory = 0.0
        
        history["train_loss"].append(avg_train_loss)
        history["dev_loss"].append(dev_loss)
        history["dev_f1"].append(dev_f1)
        history["epoch_times"].append(epoch_time)
        history["gpu_memory_gb"].append(max_memory)
        
        logger.info(f"\nEpoch {epoch+1}/{EPOCHS}:")
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
    
    # Save training history
    total_training_time = time.time() - total_training_start
    history["total_training_time_hours"] = total_training_time / 3600
    
    with open(os.path.join(SAVE_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    
    logger.info("="*60)
    logger.info("Training complete!")
    logger.info(f"Best F1: {history['best_f1']:.4f} at epoch {history['best_epoch']}")
    logger.info(f"Total training time: {total_training_time/3600:.2f} hours")
    logger.info(f"Trainable params: {param_stats['trainable']:,} ({param_stats['trainable_pct']:.2f}%)")
    logger.info(f"Model saved to {SAVE_DIR}/best_model.pt")
    logger.info(f"Training log saved to {log_file}")
    logger.info("="*60)
    
    return history


if __name__ == "__main__":
    train()
