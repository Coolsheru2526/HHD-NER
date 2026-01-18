from indicnlp.tokenize import indic_tokenize

import os
os.environ["INDIC_RESOURCES_PATH"] = r"D:\ML\NER\hindi\indic_nlp_resources"


def label_sentence(sentence, gazetteers):
    tokens = indic_tokenize.trivial_tokenize(sentence)
    labels = ["O"] * len(tokens)

    for entity, words in gazetteers.items():
        for i, tok in enumerate(tokens):
            if tok in words:
                labels[i] = f"B-{entity}"

    return list(zip(tokens, labels))

def write_conll(sentences, gazetteers, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for sent in sentences:
            tagged = label_sentence(sent, gazetteers)
            for word, tag in tagged:
                f.write(f"{word}\t{tag}\n")
            f.write("\n")
