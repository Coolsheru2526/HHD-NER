

import os
import torch
from indicnlp.tokenize import indic_tokenize

from model_baseline import MuRIL_CRF
from dataset import read_conll, encode
from evaluation import evaluate_ner, read_conll_for_eval

os.environ["INDIC_RESOURCES_PATH"] = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "indic_nlp_resources"
)

# Tag schema (must match training)
TAGS = ["O", "B-DISEASE", "I-DISEASE", "B-SYMPTOM", "I-SYMPTOM", 
        "B-CONSUMABLE", "I-CONSUMABLE", "B-PERSON", "I-PERSON"]
tag2id = {t: i for i, t in enumerate(TAGS)}
id2tag = {i: t for t, i in tag2id.items()}


def load_model(model_path: str, device: torch.device):
    """Load trained MuRIL-CRF model."""
    model = MuRIL_CRF(len(TAGS)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def predict_sentence(model, sentence: str, device: torch.device):
    """
    Run NER prediction on a single Hindi sentence.
    
    Returns:
        List of (token, predicted_tag) tuples
    """
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
    
    # Tokenize
    tokens = indic_tokenize.trivial_tokenize(sentence)
    
    # Encode
    encoding = tokenizer(
        [tokens],
        is_split_into_words=True,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    # Predict
    with torch.no_grad():
        predictions = model(input_ids, attention_mask)
    
    # Align predictions with original tokens
    word_ids = encoding.word_ids(batch_index=0)
    results = []
    prev_word_id = None
    
    for i, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        if word_id != prev_word_id:
            pred_tag = id2tag.get(predictions[0][i], "O")
            results.append((tokens[word_id], pred_tag))
        prev_word_id = word_id
    
    return results


def predict_file(model, input_path: str, output_path: str, device: torch.device):
    """
    Run predictions on a CoNLL file and save results.
    
    Args:
        model: Trained NER model
        input_path: Path to input CoNLL file
        output_path: Path for output predictions
        device: Torch device
    """
    sentences, true_labels = read_conll(input_path)
    enc = encode(sentences, true_labels, tag2id)
    
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    labels = enc["labels"]
    
    all_preds = []
    all_true = []
    
    model.eval()
    with torch.no_grad():
        for i in range(input_ids.size(0)):
            predictions = model(
                input_ids[i:i+1],
                attention_mask[i:i+1]
            )
            
            pred_seq = []
            true_seq = []
            
            for j, (pred_id, true_id) in enumerate(zip(predictions[0], labels[i].tolist())):
                if true_id != -100:
                    pred_seq.append(id2tag.get(pred_id, "O"))
                    true_seq.append(id2tag.get(true_id, "O"))
            
            all_preds.append(pred_seq)
            all_true.append(true_seq)
    
    # Write predictions
    with open(output_path, "w", encoding="utf-8") as f:
        for sent_tokens, pred_tags in zip(sentences, all_preds):
            for token, tag in zip(sent_tokens, pred_tags):
                f.write(f"{token}\t{tag}\n")
            f.write("\n")
    
    print(f"[predict] Predictions saved to {output_path}")
    
    return all_true, all_preds


def run_evaluation(model_path: str, test_path: str):
    """
    Complete evaluation pipeline.
    Loads model, runs predictions, computes metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[predict] Using device: {device}")
    
    # Load model
    print(f"[predict] Loading model from {model_path}")
    model = load_model(model_path, device)
    
    # Run predictions
    output_path = test_path.replace(".conll", "_predictions.conll")
    true_labels, pred_labels = predict_file(model, test_path, output_path, device)
    
    # Evaluate
    metrics = evaluate_ner(true_labels, pred_labels, 
                          output_path.replace(".conll", "_report.txt"))
    
    return metrics


def demo_interactive():
    """Interactive demo mode for testing individual sentences."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "weights/best_model.pt"
    
    if not os.path.exists(model_path):
        print(f"[Error] Model not found at {model_path}")
        print("Please train the model first using train_baseline.py")
        return
    
    model = load_model(model_path, device)
    print("[predict] Model loaded. Enter Hindi sentences (or 'quit' to exit):\n")
    
    while True:
        sentence = input(">>> ").strip()
        if sentence.lower() in ['quit', 'exit', 'q']:
            break
        
        if not sentence:
            continue
        
        results = predict_sentence(model, sentence, device)
        print("\nPredictions:")
        for token, tag in results:
            if tag != "O":
                print(f"  {token:20s} -> {tag}")
            else:
                print(f"  {token:20s}    {tag}")
        print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python predict.py demo                              - Interactive demo (baseline)")
        print("  python predict.py <test_file>                       - Evaluate baseline on test set")
        print("  python predict.py <test_file> <model_path>          - Evaluate custom model")
        print("\nExamples:")
        print("  python predict.py data/processed/test.conll")
        print("  python predict.py data/processed/test.conll weights/frozen_encoder/best_model.pt")
        print("\nRunning interactive demo...")
        demo_interactive()
    
    elif sys.argv[1] == "demo":
        demo_interactive()
    
    else:
        # Get test file path
        test_path = sys.argv[1]
        
        # Get model path (default to baseline if not specified)
        if len(sys.argv) >= 3:
            model_path = sys.argv[2]
        else:
            model_path = "weights/best_model.pt"
        
        print(f"[predict] Testing with model: {model_path}")
        run_evaluation(model_path, test_path)

