
import os
import random
from typing import List, Tuple


def read_conll_sentences(path: str) -> List[List[Tuple[str, str]]]:
    sentences = []
    current_sentence = []
    
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            
            if not line:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                continue
            
            parts = line.rsplit(maxsplit=1)
            if len(parts) == 2:
                current_sentence.append((parts[0], parts[1]))
    
    # Handle last sentence
    if current_sentence:
        sentences.append(current_sentence)
    
    return sentences


def write_conll_sentences(sentences: List[List[Tuple[str, str]]], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        for sentence in sentences:
            for token, tag in sentence:
                f.write(f"{token}\t{tag}\n")
            f.write("\n")


def split_data(
    input_path: str,
    output_dir: str = "data/processed",
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> dict:
    """
    Split CoNLL data into train/dev/test sets.
    
    Args:
        input_path: Path to input CoNLL file
        output_dir: Directory for output files
        train_ratio: Proportion for training (default 0.8)
        dev_ratio: Proportion for validation (default 0.1)
        test_ratio: Proportion for testing (default 0.1)
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with file paths and split statistics
    """
    assert abs(train_ratio + dev_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # Read and shuffle sentences
    sentences = read_conll_sentences(input_path)
    random.seed(seed)
    random.shuffle(sentences)
    
    total = len(sentences)
    train_end = int(total * train_ratio)
    dev_end = train_end + int(total * dev_ratio)
    
    train_data = sentences[:train_end]
    dev_data = sentences[train_end:dev_end]
    test_data = sentences[dev_end:]
    
    # Write splits
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, "train_split.conll")
    dev_path = os.path.join(output_dir, "dev.conll")
    test_path = os.path.join(output_dir, "test.conll")
    
    write_conll_sentences(train_data, train_path)
    write_conll_sentences(dev_data, dev_path)
    write_conll_sentences(test_data, test_path)
    
    stats = {
        "total": total,
        "train": len(train_data),
        "dev": len(dev_data),
        "test": len(test_data),
        "train_path": train_path,
        "dev_path": dev_path,
        "test_path": test_path
    }
    
    print(f"[split_data] Split {total} sentences:")
    print(f"  Train: {stats['train']} ({train_ratio*100:.0f}%) -> {train_path}")
    print(f"  Dev:   {stats['dev']} ({dev_ratio*100:.0f}%) -> {dev_path}")
    print(f"  Test:  {stats['test']} ({test_ratio*100:.0f}%) -> {test_path}")
    
    return stats


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        # Default: split the v2 conll file
        input_file = "data/processed/train_v2.conll"
    else:
        input_file = sys.argv[1]
    
    split_data(input_file)
