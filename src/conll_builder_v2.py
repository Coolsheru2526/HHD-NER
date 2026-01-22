
import os
from indicnlp.tokenize import indic_tokenize
from indicnlp.morph import unsupervised_morph

# Set Indic NLP resources path
os.environ["INDIC_RESOURCES_PATH"] = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "indic_nlp_resources"
)

# Initialize Hindi morphological analyzer for stemming
try:
    morph_analyzer = unsupervised_morph.UnsupervisedMorphAnalyzer('hi')
except Exception:
    morph_analyzer = None
    print("[Warning] Morphological analyzer not available. Using exact matching only.")


def stem_word(word: str) -> str:
    """
    Extract morphological root of a Hindi word.
    Falls back to original word if stemming fails.
    """
    if morph_analyzer is None:
        return word.lower()
    try:
        # Get the first (most likely) morphological analysis
        analyses = morph_analyzer.morph_analyze(word)
        if analyses:
            return analyses[0].lower()
        return word.lower()
    except Exception:
        return word.lower()


def load_gazetteer_with_stems(path: str) -> dict:
    exact_phrases = set()
    stemmed_to_original = {}
    
    with open(path, encoding="utf-8") as f:
        for line in f:
            phrase = line.strip()
            if not phrase:
                continue
            
            # Store exact phrase
            exact_phrases.add(phrase.lower())
            
            # Create stemmed version
            tokens = phrase.split()
            stemmed_tokens = [stem_word(t) for t in tokens]
            stemmed_phrase = " ".join(stemmed_tokens)
            stemmed_to_original[stemmed_phrase] = phrase
    
    return {
        "exact": exact_phrases,
        "stemmed": stemmed_to_original
    }


def label_sentence_advanced(sentence: str, gazetteers: dict, max_window: int = 4) -> list:
    """  
    Algorithm:
    1. Tokenize sentence
    2. For window sizes [max_window, ..., 1], check if token window matches gazetteer
    3. Match longest phrase first (greedy)
    4. Assign B- to first token, I- to subsequent tokens
    5. Two passes: exact match first, then stemmed match for remaining O tokens
    
    Args:
        sentence: Raw Hindi sentence
        gazetteers: Dict mapping entity_type -> {"exact": set, "stemmed": dict}
        max_window: Maximum phrase length to consider
    
    Returns:
        List of (token, tag) tuples
    """
    tokens = indic_tokenize.trivial_tokenize(sentence)
    labels = ["O"] * len(tokens)
    
    # Prepare token representations
    tokens_lower = [t.lower() for t in tokens]
    tokens_stemmed = [stem_word(t) for t in tokens]
    
    def try_match(token_list: list, gazetteer_set_or_dict, is_stemmed: bool):
        """Try to match tokens against gazetteer using greedy longest-match."""
        nonlocal labels
        
        # Iterate window sizes from largest to smallest
        for window_size in range(max_window, 0, -1):
            i = 0
            while i <= len(tokens) - window_size:
                # Skip if any token in window is already labeled
                if any(labels[j] != "O" for j in range(i, i + window_size)):
                    i += 1
                    continue
                
                # Build phrase from window
                window_tokens = token_list[i:i + window_size]
                phrase = " ".join(window_tokens)
                
                # Check match
                if is_stemmed:
                    matched = phrase in gazetteer_set_or_dict
                else:
                    matched = phrase in gazetteer_set_or_dict
                
                if matched:
                    # Found match - apply BIO tags
                    labels[i] = f"B-{entity_type}"
                    for j in range(1, window_size):
                        labels[i + j] = f"I-{entity_type}"
                    i += window_size  # Skip past matched tokens
                else:
                    i += 1
    
    # Process each entity type
    for entity_type, gaz_data in gazetteers.items():
        # Pass 1: Exact matching (high precision)
        try_match(tokens_lower, gaz_data["exact"], is_stemmed=False)
        
        # Pass 2: Stemmed matching (high recall) for remaining O tokens
        try_match(tokens_stemmed, gaz_data["stemmed"], is_stemmed=True)
    
    return list(zip(tokens, labels))


def write_conll_advanced(sentences: list, gazetteers: dict, out_path: str, max_window: int = 4):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    with open(out_path, "w", encoding="utf-8") as f:
        for sent in sentences:
            tagged = label_sentence_advanced(sent, gazetteers, max_window)
            for word, tag in tagged:
                f.write(f"{word}\t{tag}\n")
            f.write("\n")  
    print(f"[conll_builder_v2] Wrote {len(sentences)} sentences to {out_path}")


def build_conll_v2(
    raw_data_path: str = "data/raw/Hindi_Health_Data.txt",
    output_path: str = "data/processed/train_v2.conll",
    max_window: int = 4
):
    from preprocess import extract_hindi_sentences
    
    # Load gazetteers with stemming support
    gazetteers = {
        "DISEASE": load_gazetteer_with_stems("data/raw/Disease_Gazetteer.txt"),
        "SYMPTOM": load_gazetteer_with_stems("data/raw/Symptom_Gazetteer.txt"),
        "CONSUMABLE": load_gazetteer_with_stems("data/raw/Consumable_Gazetteer.txt"),
        "PERSON": load_gazetteer_with_stems("data/raw/Person_Gazetteer.txt"),
    }
    
    # Extract clean Hindi sentences
    sentences = extract_hindi_sentences(raw_data_path)
    print(f"[conll_builder_v2] Extracted {len(sentences)} Hindi sentences")
    
    # Write advanced CoNLL file
    write_conll_advanced(sentences, gazetteers, output_path, max_window)


if __name__ == "__main__":
    build_conll_v2()
