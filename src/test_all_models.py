import os
import sys
import json
import torch
import numpy as np
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score
)
from seqeval.scheme import IOB2

from dataset import read_conll, encode
from model_baseline import MuRIL_CRF
from model_softmax import MuRIL_Softmax
from model_lora import MuRIL_LoRA_CRF
from model_stacked import MuRIL_NER  # For the Stacked model


# ================= TAGS =================
TAGS = [
    "O",
    "B-DISEASE", "I-DISEASE",
    "B-SYMPTOM", "I-SYMPTOM",
    "B-CONSUMABLE", "I-CONSUMABLE",
    "B-PERSON", "I-PERSON"
]

tag2id = {t: i for i, t in enumerate(TAGS)}
id2tag = {i: t for t, i in tag2id.items()}


# ================= MODEL LOADER =================
def load_model_by_type(model_path, model_type, device):
    """Load model based on experiment type."""
    if model_type == "no_crf":
        model = MuRIL_Softmax(len(TAGS)).to(device)
    elif model_type == "lora":
        model = MuRIL_LoRA_CRF(len(TAGS)).to(device)
    elif model_type == "advanced" or model_type == "stacked":
        model = MuRIL_NER(len(TAGS)).to(device)
    else:
        # baseline or frozen encoder both use MuRIL_CRF class
        model = MuRIL_CRF(len(TAGS)).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model


# ================= SINGLE MODEL TEST =================
def test_model(model_path, test_path, model_type, device):
    """Test a single model on test set."""
    model = load_model_by_type(model_path, model_type, device)

    # Load test data
    sentences, true_labels = read_conll(test_path)
    enc = encode(sentences, true_labels, tag2id)

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    labels = enc["labels"]  # kept on CPU, fine

    all_preds = []
    all_true = []

    model.eval()
    with torch.no_grad():
        for i in range(input_ids.size(0)):

            predictions = model(
                input_ids[i:i + 1],
                attention_mask[i:i + 1]
            )

            # Standard models (CRF/Softmax/Advanced) all return a list-like output in inference mode
            pred_ids = predictions[0]

            pred_seq = []
            true_seq = []

            for pred_id, true_id in zip(pred_ids, labels[i].tolist()):
                if true_id != -100:
                    pred_seq.append(id2tag.get(pred_id, "O"))
                    true_seq.append(id2tag.get(true_id, "O"))

            all_preds.append(pred_seq)
            all_true.append(true_seq)

    # Overall metrics (strict entity evaluation with IOB2)
    precision = precision_score(all_true, all_preds, mode="strict", scheme=IOB2)
    recall = recall_score(all_true, all_preds, mode="strict", scheme=IOB2)
    f1 = f1_score(all_true, all_preds, mode="strict", scheme=IOB2)
    report = classification_report(all_true, all_preds, mode="strict", scheme=IOB2)

    return precision, recall, f1, report


# ================= ALL MODELS TEST =================
def test_all_models():
    """Test all trained models across all 3 seeds and report Mean ± Std Dev."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    test_path = "data/processed/test.conll"
    weights_root = sys.argv[1] if len(sys.argv) > 1 else "weights"

    if not os.path.exists(test_path):
        print(f"Error: Test file not found at {test_path}")
        print("Please run: python src/split_data.py")
        return

    SEEDS = [42, 123, 2024]
    
    # User specified 5 models: Baseline, LoRA, No-CRF, Frozen Encoder, Advanced Stacked
    experiments = [
        {
            "name": "Line 1: Baseline (MuRIL + CRF)",
            "model_type": "baseline",
            "dir_name": "baseline_crf"
        },
        {
            "name": "Line 2: Frozen Encoder (CRF only)",
            "model_type": "frozen",
            "dir_name": "frozen_encoder"
        },
        {
            "name": "Line 3: No-CRF (Softmax)",
            "model_type": "no_crf",
            "dir_name": "no_crf"
        },
        {
            "name": "Line 4: LoRA Fine-tuning",
            "model_type": "lora",
            "dir_name": "lora"
        },
        {
            "name": "Line 5: Stacked (Advanced)",
            "model_type": "advanced",
            "dir_name": "advanced_model"
        }
    ]

    all_results = []

    print("=" * 80)
    print("TESTING 5 MODELS ACROSS 3 SEEDS ON TEST SET")
    print("=" * 80)
    print()

    for exp in experiments:
        exp_name = exp["name"]
        exp_dir = exp["dir_name"]
        
        seed_precisions = []
        seed_recalls = []
        seed_f1s = []
        
        print(f"--- Experiment: {exp_name} ---")

        for seed in SEEDS:
            # Construct path according to laboratory PC structure
            model_path = os.path.join(weights_root, exp_dir, f"seed_{seed}", "best_model.pt")

            if not os.path.exists(model_path):
                print(f"  ⚠️  Seed {seed}: Model not found at {model_path}")
                continue

            print(f"  Testing Seed {seed}...")
            
            try:
                prec, rec, f1, _ = test_model(
                    model_path=model_path,
                    test_path=test_path,
                    model_type=exp["model_type"],
                    device=device
                )
                
                seed_precisions.append(prec)
                seed_recalls.append(rec)
                seed_f1s.append(f1)
                
                print(f"    P: {prec:.4f} | R: {rec:.4f} | F1: {f1:.4f}")
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
                continue
        
        # Calculate statistics if we have data
        if seed_f1s:
            results = {
                "name": exp_name,
                "precision": {"mean": np.mean(seed_precisions), "std": np.std(seed_precisions), "all": seed_precisions},
                "recall": {"mean": np.mean(seed_recalls), "std": np.std(seed_recalls), "all": seed_recalls},
                "f1": {"mean": np.mean(seed_f1s), "std": np.std(seed_f1s), "all": seed_f1s}
            }
            all_results.append(results)
            print(f"  Done. Aggregated Outcome:")
            print(f"    F1-Score  : {results['f1']['mean']:.4f} ± {results['f1']['std']:.4f}")
            print(f"    Precision : {results['precision']['mean']:.4f} ± {results['precision']['std']:.4f}")
            print(f"    Recall    : {results['recall']['mean']:.4f} ± {results['recall']['std']:.4f}\n")
        else:
            print(f"  ❌ No seeds found for {exp_name}\n")

    # ================= SUMMARY TABLE =================
    if all_results:
        print("\n" + "=" * 90)
        print("FINAL TEST SET SUMMARY (Mean ± Standard Deviation)")
        print("=" * 90)
        print()
        print(f"{'Experiment Architecture':<35} {'Precision':<15} {'Recall':<15} {'F1-Score':<15}")
        print("-" * 90)

        for r in all_results:
            p_str = f"{r['precision']['mean']:.4f} ± {r['precision']['std']:.4f}"
            r_str = f"{r['recall']['mean']:.4f} ± {r['recall']['std']:.4f}"
            f_str = f"{r['f1']['mean']:.4f} ± {r['f1']['std']:.4f}"
            
            print(
                f"{r['name']:<35} "
                f"{p_str:<15} "
                f"{r_str:<15} "
                f"{f_str:<15}"
            )

        print()
        print("=" * 90)

        # Save summary JSON
        summary_dir = os.path.join(weights_root, "final_test_eval")
        os.makedirs(summary_dir, exist_ok=True)

        summary_file = os.path.join(summary_dir, "test_summary_stats.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

        print(f"Aggregated summary saved to: {summary_file}")

    else:
        print("No models were successfully tested.")


if __name__ == "__main__":
    test_all_models()
