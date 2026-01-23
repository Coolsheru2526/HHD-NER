
import os
import json
import torch
from seqeval.metrics import classification_report, f1_score
from seqeval.scheme import IOB2
from dataset import read_conll, encode
from model import MuRIL_CRF
from model_softmax import MuRIL_Softmax
from model_lora import MuRIL_LoRA_CRF

TAGS = ["O", "B-DISEASE", "I-DISEASE", "B-SYMPTOM", "I-SYMPTOM", 
        "B-CONSUMABLE", "I-CONSUMABLE", "B-PERSON", "I-PERSON"]
tag2id = {t: i for i, t in enumerate(TAGS)}
id2tag = {i: t for t, i in tag2id.items()}


def load_model_by_type(model_path, model_type, device):
    """Load model based on experiment type."""
    if model_type == "no_crf":
        model = MuRIL_Softmax(len(TAGS)).to(device)
    elif model_type == "lora":
        model = MuRIL_LoRA_CRF(len(TAGS)).to(device)
    else:  # baseline or frozen (both use MuRIL_CRF)
        model = MuRIL_CRF(len(TAGS)).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def test_model(model_path, test_path, model_type, device):
    """Test a single model on test set."""
    
    # Load model
    model = load_model_by_type(model_path, model_type, device)
    
    # Load test data
    sentences, true_labels = read_conll(test_path)
    enc = encode(sentences, true_labels, tag2id)
    
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    labels = enc["labels"]
    
    # Get predictions
    all_preds = []
    all_true = []
    
    model.eval()
    with torch.no_grad():
        for i in range(input_ids.size(0)):
            predictions = model(
                input_ids[i:i+1],
                attention_mask[i:i+1]
            )
            
            if model_type == "no_crf":
                # Softmax returns list of lists
                pred_ids = predictions[0]
            else:
                # CRF returns list
                pred_ids = predictions[0]
            
            pred_seq = []
            true_seq = []
            
            for j, (pred_id, true_id) in enumerate(zip(pred_ids, labels[i].tolist())):
                if true_id != -100:
                    pred_seq.append(id2tag.get(pred_id, "O"))
                    true_seq.append(id2tag.get(true_id, "O"))
            
            all_preds.append(pred_seq)
            all_true.append(true_seq)
    
    # Compute metrics
    f1 = f1_score(all_true, all_preds, mode='strict', scheme=IOB2)
    report = classification_report(all_true, all_preds, mode='strict', scheme=IOB2)
    
    return f1, report


def test_all_models():
    """Test all trained models on test set."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    test_path = "data/processed/test.conll"
    
    if not os.path.exists(test_path):
        print(f"Error: Test file not found at {test_path}")
        print("Please run: python src/split_data.py")
        return
    
    experiments = [
        {
            "name": "Baseline (MuRIL + CRF)",
            "model_path": "weights/best_model.pt",
            "model_type": "baseline"
        },
        {
            "name": "Frozen Encoder (CRF only)",
            "model_path": "weights/frozen_encoder/best_model.pt",
            "model_type": "frozen"
        },
        {
            "name": "No-CRF (Softmax)",
            "model_path": "weights/no_crf/best_model.pt",
            "model_type": "no_crf"
        },
        {
            "name": "LoRA Fine-tuning",
            "model_path": "weights/lora/best_model.pt",
            "model_type": "lora"
        }
    ]
    
    results = []
    
    print("="*80)
    print("TESTING ALL MODELS ON TEST SET")
    print("="*80)
    print()
    
    for exp in experiments:
        model_path = exp["model_path"]
        
        if not os.path.exists(model_path):
            print(f"⚠️  {exp['name']}: Model not found")
            print(f"   Expected at: {model_path}")
            print()
            continue
        
        print(f"Testing: {exp['name']}")
        print(f"Model: {model_path}")
        
        try:
            f1, report = test_model(model_path, test_path, exp["model_type"], device)
            
            print(f"Test F1: {f1:.4f}")
            print()
            
            results.append({
                "name": exp["name"],
                "f1": f1,
                "report": report
            })
            
            # Save detailed report
            report_file = model_path.replace("best_model.pt", "test_report.txt")
            with open(report_file, 'w') as f:
                f.write(f"Test Set Evaluation - {exp['name']}\n")
                f.write("="*80 + "\n\n")
                f.write(f"Test F1: {f1:.4f}\n\n")
                f.write("Per-Entity Metrics:\n")
                f.write(report)
            
            print(f"   Detailed report saved: {report_file}")
            print()
            
        except Exception as e:
            print(f"   Error: {e}")
            print()
            continue
    
    # Print summary
    if results:
        print("="*80)
        print("TEST SET RESULTS SUMMARY")
        print("="*80)
        print()
        print(f"{'Model':<35} {'Test F1':<10}")
        print("-"*80)
        
        for r in results:
            print(f"{r['name']:<35} {r['f1']:<10.4f}")
        
        print()
        print("="*80)
        
        # Save summary JSON
        summary = {
            "test_set": test_path,
            "results": [{"model": r["name"], "f1": r["f1"]} for r in results]
        }
        
        with open("weights/test_results_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("Summary saved to: weights/test_results_summary.json")
    
    else:
        print("No models were successfully tested.")


if __name__ == "__main__":
    test_all_models()
