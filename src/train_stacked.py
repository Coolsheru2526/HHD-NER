
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
from model_stacked import MuRIL_NER
from evaluation import validate_bio_sequences, compute_entity_distribution


DATA_DIR = "data/processed"
TRAIN_PATH = os.path.join(DATA_DIR, "train_split.conll")
DEV_PATH = os.path.join(DATA_DIR, "dev.conll")
BASE_SAVE_DIR = "weights/advanced_model"  # Changed from SAVE_DIR to BASE_SAVE_DIR

# Hyperparameters
EPOCHS = 15  # Sufficient for SOTA run
BATCH_SIZE = 16
PATIENCE = 3

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


def load_dataset(path, tag2id):
    sentences, labels = read_conll(path)
    enc = encode(sentences, labels, tag2id)
    return NERDataset(enc["input_ids"], enc["attention_mask"], enc["labels"])


def validate(model, dataloader, device):
    """Run validation and compute Precision, Recall, and F1."""
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
                for pred_id, true_id in zip(predictions[i], labels[i].cpu().tolist()):
                    if true_id != -100:
                        pred_seq.append(id2tag[pred_id])
                        true_seq.append(id2tag[true_id])
                all_preds.append(pred_seq)
                all_labels.append(true_seq)

    avg_loss = total_loss / len(dataloader)

    from seqeval.metrics import f1_score, precision_score, recall_score
    from seqeval.scheme import IOB2
    
    f1 = f1_score(all_labels, all_preds, mode="strict", scheme=IOB2)
    precision = precision_score(all_labels, all_preds, mode="strict", scheme=IOB2)
    recall = recall_score(all_labels, all_preds, mode="strict", scheme=IOB2)

    model.train()
    return avg_loss, f1, precision, recall


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "trainable_pct": 100 * trainable / total}


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(seed, save_dir):
    """Main training function for the advanced stacked model."""
    
    set_seed(seed)
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup logging
    log_file = os.path.join(save_dir, "training.log")
    
    # Reset logging handlers to avoid mixing logs from different seeds
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

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
    logger.info(f"ADVANCED MODEL: MuRIL + BiLSTM + Attention + CNN + CRF - SEED: {seed}")
    logger.info("="*60)
    logger.info(f"Using device: {device}")
    logger.info(f"Random seed: {seed}")
    logger.info(f"Training log: {log_file}")

    # Validate data
    logger.info("Validating training data...")
    _, train_labels_check = read_conll(TRAIN_PATH)
    validation = validate_bio_sequences(train_labels_check)

    if not validation["is_valid"]:
        logger.error(f"Found {len(validation['violations'])} BIO violations!")
        sys.exit(1)

    logger.info("[OK] BIO sequences validated")

    # Entity distribution
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
    logger.info(f"Batch size: {BATCH_SIZE}, Epochs: {EPOCHS}")

    # Initialize model
    logger.info("Initializing advanced model...")
    model = MuRIL_NER(len(TAGS)).to(device)

    # Count parameters
    param_stats = count_parameters(model)
    logger.info("Model parameters:")
    logger.info(f"  Total:     {param_stats['total']:,}")
    logger.info(f"  Trainable: {param_stats['trainable']:,} ({param_stats['trainable_pct']:.2f}%)")

    # Optimizer with different learning rates for different components
    optimizer = torch.optim.AdamW([
        {"params": model.muril.parameters(), "lr": 2e-5},      # Low LR for pretrained
        {"params": model.bilstm.parameters(), "lr": 1e-3},     # Higher for BiLSTM
        {"params": model.attention.parameters(), "lr": 1e-3},  # Higher for attention
        {"params": model.cnns.parameters(), "lr": 1e-3},       # Higher for CNN
        {"params": model.fc.parameters(), "lr": 1e-3},         # Higher for FC
        {"params": model.crf.parameters(), "lr": 5e-4},        # Medium for CRF
    ])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, verbose=True
    )

    history = {
        "experiment": "advanced_model",
        "train_loss": [],
        "dev_loss": [],
        "dev_f1": [],
        "dev_precision": [],
        "dev_recall": [],
        "epoch_times": [],
        "gpu_memory_gb": [],
        "best_f1": 0.0,
        "best_precision": 0.0,
        "best_recall": 0.0,
        "best_epoch": 0,
        "seed": seed,
        "batch_size": BATCH_SIZE,
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

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = epoch_loss / len(train_loader)

        # Validation
        dev_loss, dev_f1, dev_p, dev_r = validate(model, dev_loader, device)

        # Update scheduler
        scheduler.step(dev_f1)
        epoch_time = time.time() - epoch_start_time

        if torch.cuda.is_available():
            max_memory = torch.cuda.max_memory_allocated() / (1024**3)
            torch.cuda.reset_peak_memory_stats()
        else:
            max_memory = 0.0

        history["train_loss"].append(avg_train_loss)
        history["dev_loss"].append(dev_loss)
        history["dev_f1"].append(dev_f1)
        history["dev_precision"].append(dev_p)
        history["dev_recall"].append(dev_r)
        history["epoch_times"].append(epoch_time)
        history["gpu_memory_gb"].append(max_memory)

        logger.info(f"\nEpoch {epoch+1}/{EPOCHS}:")
        logger.info(f"  Train Loss: {avg_train_loss:.4f}")
        logger.info(f"  Dev Loss:   {dev_loss:.4f}")
        logger.info(f"  Dev P:      {dev_p:.4f}, R: {dev_r:.4f}, F1: {dev_f1:.4f}")
        logger.info(f"  Time:       {epoch_time/60:.1f} min")
        logger.info(f"  GPU Memory: {max_memory:.2f} GB")

        # Save best model
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            history["best_f1"] = best_f1
            history["best_precision"] = dev_p
            history["best_recall"] = dev_r
            history["best_epoch"] = epoch + 1
            patience_counter = 0

            model_path = os.path.join(save_dir, "best_model.pt")
            torch.save(model.state_dict(), model_path)
            logger.info(f"  [OK] New best model saved (F1={best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(f"\nEarly stopping at epoch {epoch+1} (patience={PATIENCE})")
                break

    # Save training history
    total_training_time = time.time() - total_training_start
    history["total_training_time_hours"] = total_training_time / 3600

    with open(os.path.join(save_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    logger.info("="*60)
    logger.info("Training complete!")
    logger.info(f"Best metrics at epoch {history['best_epoch']}:")
    logger.info(f"  F1: {history['best_f1']:.4f}")
    logger.info(f"  P:  {history['best_precision']:.4f}")
    logger.info(f"  R:  {history['best_recall']:.4f}")
    logger.info(f"Total training time: {total_training_time/3600:.2f} hours")
    logger.info(f"Model saved to {save_dir}/best_model.pt")
    logger.info(f"Training log saved to {log_file}")
    logger.info("="*60)

    return history


if __name__ == "__main__":
    seeds = [42, 123, 2024]
    all_f1 = []
    all_precision = []
    all_recall = []
    
    # Ensure base directory exists
    os.makedirs(BASE_SAVE_DIR, exist_ok=True)
    
    print("\nStarting Advanced SOTA statistical robustness experiment with 3 seeds...")
    print("="*60)
    
    for i, seed in enumerate(seeds):
        current_save_dir = os.path.join(BASE_SAVE_DIR, f"seed_{seed}")
        print(f"\n[RUN {i+1}/3] Training with Seed: {seed}")
        
        history = train(seed, current_save_dir)
        all_f1.append(history["best_f1"])
        all_precision.append(history["best_precision"])
        all_recall.append(history["best_recall"])
        
    # Aggregate results
    def get_stats(data):
        return {"mean": float(np.mean(data)), "std": float(np.std(data))}

    stats_f1 = get_stats(all_f1)
    stats_p = get_stats(all_precision)
    stats_r = get_stats(all_recall)
    
    summary = {
        "experiment": "advanced_sota_robustness",
        "seeds": seeds,
        "metrics": {
            "f1": {"all": all_f1, **stats_f1},
            "precision": {"all": all_precision, **stats_p},
            "recall": {"all": all_recall, **stats_r}
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    summary_path = os.path.join(BASE_SAVE_DIR, "summary_stats.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
        
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE!")
    print(f"F1-score:  {stats_f1['mean']:.4f} ± {stats_f1['std']:.4f}")
    print(f"Precision: {stats_p['mean']:.4f} ± {stats_p['std']:.4f}")
    print(f"Recall:    {stats_r['mean']:.4f} ± {stats_r['std']:.4f}")
    print(f"Summary saved to: {summary_path}")
    print("="*60)
