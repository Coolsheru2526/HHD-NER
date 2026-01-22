
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score
)
from seqeval.scheme import IOB2
from collections import defaultdict


def read_conll_for_eval(path: str) -> tuple:
    sentences, labels = [], []
    current_tokens, current_labels = [], []
    
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            
            if not line:
                if current_tokens:
                    sentences.append(current_tokens)
                    labels.append(current_labels)
                    current_tokens, current_labels = [], []
                continue
            
            parts = line.rsplit(maxsplit=1)
            if len(parts) == 2:
                token, label = parts
                current_tokens.append(token)
                current_labels.append(label)
    
    # Handle last sentence if file doesn't end with newline
    if current_tokens:
        sentences.append(current_tokens)
        labels.append(current_labels)
    
    return sentences, labels


def evaluate_ner(
    true_labels: list,
    pred_labels: list,
    output_path: str = None
) -> dict:
    """
    Comprehensive NER evaluation with seqeval.
    
    Args:
        true_labels: List of label sequences (ground truth)
        pred_labels: List of label sequences (predictions)
        output_path: Optional path to save detailed report
    
    Returns:
        Dictionary with metrics
    """
    # Calculate metrics
    metrics = {
        "f1_micro": f1_score(true_labels, pred_labels, mode='strict', scheme=IOB2),
        "precision_micro": precision_score(true_labels, pred_labels, mode='strict', scheme=IOB2),
        "recall_micro": recall_score(true_labels, pred_labels, mode='strict', scheme=IOB2),
        "f1_macro": f1_score(true_labels, pred_labels, mode='strict', scheme=IOB2, average='macro'),
    }
    
    # Generate detailed classification report
    report = classification_report(true_labels, pred_labels, mode='strict', scheme=IOB2)
    
    print("\n" + "="*60)
    print("NER EVALUATION RESULTS")
    print("="*60)
    print(report)
    print(f"\nMicro F1:  {metrics['f1_micro']:.4f}")
    print(f"Macro F1:  {metrics['f1_macro']:.4f}")
    print("="*60 + "\n")
    
    # Save to file if path provided
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("NER EVALUATION RESULTS\n")
            f.write("="*60 + "\n")
            f.write(report)
            f.write(f"\nMicro F1:  {metrics['f1_micro']:.4f}\n")
            f.write(f"Macro F1:  {metrics['f1_macro']:.4f}\n")
        print(f"Report saved to {output_path}")
    
    return metrics


def compute_entity_distribution(labels: list) -> dict:
    """
    Analyze entity type distribution in dataset.
    Useful for understanding class imbalance.
    
    Args:
        labels: List of label sequences
    
    Returns:
        Dictionary with entity counts
    """
    entity_counts = defaultdict(int)
    
    for label_seq in labels:
        for label in label_seq:
            if label.startswith("B-"):
                entity_type = label[2:]  # Remove B- prefix
                entity_counts[entity_type] += 1
    
    return dict(entity_counts)


def validate_bio_sequences(labels: list) -> dict:
    """
    Check for BIO tagging violations.
    Reports orphaned I- tags (I- without preceding B-).
    
    Args:
        labels: List of label sequences
    
    Returns:
        Dictionary with validation results
    """
    violations = []
    total_i_tags = 0
    valid_i_tags = 0
    
    for sent_idx, label_seq in enumerate(labels):
        prev_label = "O"
        for tok_idx, label in enumerate(label_seq):
            if label.startswith("I-"):
                total_i_tags += 1
                entity_type = label[2:]
                
                # Valid: I-X follows B-X or I-X
                if prev_label == f"B-{entity_type}" or prev_label == f"I-{entity_type}":
                    valid_i_tags += 1
                else:
                    violations.append({
                        "sentence": sent_idx,
                        "position": tok_idx,
                        "label": label,
                        "prev_label": prev_label
                    })
            
            prev_label = label
    
    return {
        "total_i_tags": total_i_tags,
        "valid_i_tags": valid_i_tags,
        "violations": violations,
        "is_valid": len(violations) == 0
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python evaluation.py <conll_file>")
        print("       python evaluation.py <true_file> <pred_file>")
        sys.exit(1)
    
    # Single file: analyze distribution and validate BIO
    if len(sys.argv) == 2:
        sentences, labels = read_conll_for_eval(sys.argv[1])
        
        print(f"Loaded {len(sentences)} sentences")
        
        # Entity distribution
        dist = compute_entity_distribution(labels)
        print("\nEntity Distribution:")
        for entity, count in sorted(dist.items(), key=lambda x: -x[1]):
            print(f"  {entity}: {count}")
        
        # BIO validation
        validation = validate_bio_sequences(labels)
        if validation["is_valid"]:
            print("\n✓ All BIO sequences are valid")
        else:
            print(f"\n✗ Found {len(validation['violations'])} BIO violations")
            for v in validation["violations"][:5]:
                print(f"  Sent {v['sentence']}, Pos {v['position']}: {v['label']} after {v['prev_label']}")
    
    # Two files: compare ground truth vs predicted
    elif len(sys.argv) >= 3:
        _, true_labels = read_conll_for_eval(sys.argv[1])
        _, pred_labels = read_conll_for_eval(sys.argv[2])
        
        evaluate_ner(true_labels, pred_labels)
