import os
import unicodedata
from indicnlp.tokenize import indic_tokenize


def normalize(text: str) -> str:
    text = text.replace("﻿", "")
    text = text.strip()
    text = unicodedata.normalize("NFKC", text)
    return text.lower()


def load_gazetteer(path: str) -> set:
    entries = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = normalize(line)
            if line:
                entries.add(line)
    return entries


# ---------- LABELING ----------

def label_sentence(sentence: str, gazetteers: dict, max_window=5):
    tokens = indic_tokenize.trivial_tokenize(sentence)
    tokens_norm = [normalize(t) for t in tokens]

    labels = ["O"] * len(tokens)

    for entity_type, gaz in gazetteers.items():
        for window in range(max_window, 0, -1):
            i = 0
            while i <= len(tokens) - window:
                if any(labels[j] != "O" for j in range(i, i + window)):
                    i += 1
                    continue

                phrase = " ".join(tokens_norm[i:i + window])
                if phrase in gaz:
                    labels[i] = f"B-{entity_type}"
                    for j in range(1, window):
                        labels[i + j] = f"I-{entity_type}"
                    i += window
                else:
                    i += 1

    return list(zip(tokens, labels))


def write_conll(sentences, gazetteers, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for sent in sentences:
            tagged = label_sentence(sent, gazetteers)
            for w, t in tagged:
                f.write(f"{w}\t{t}\n")
            f.write("\n")


def build_conll(
    raw_data_path="data/raw/Hindi_Health_Data.txt",
    output_path="data/processed/train_v2.conll"
):
    from preprocess import extract_hindi_sentences

    gazetteers = {
        "DISEASE": load_gazetteer("data/raw/Disease_Gazetteer.txt"),
        "SYMPTOM": load_gazetteer("data/raw/Symptom_Gazetteer.txt"),
        "CONSUMABLE": load_gazetteer("data/raw/Consumable_Gazetteer.txt"),
        "PERSON": load_gazetteer("data/raw/Person_Gazetteer.txt"),
    }

    sentences = extract_hindi_sentences(raw_data_path)
    write_conll(sentences, gazetteers, output_path)
    print(f"[OK] Wrote {len(sentences)} sentences to {output_path}")


if __name__ == "__main__":
    build_conll()
