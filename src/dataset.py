import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")

def read_conll(path):
    sentences, labels = [], []
    words, tags = [], []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                if words:
                    sentences.append(words)
                    labels.append(tags)
                    words, tags = [], []
                continue

            word, tag = line.rsplit(maxsplit=1)
            words.append(word)
            tags.append(tag)

    if words:
        sentences.append(words)
        labels.append(tags)

    return sentences, labels


def encode(sentences, labels, tag2id):
    encodings = tokenizer(
        sentences,
        is_split_into_words=True,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    encoded_labels = []
    for i, label in enumerate(labels):
        word_ids = encodings.word_ids(batch_index=i)
        prev = None
        label_ids = []
        for w in word_ids:
            if w is None:
                label_ids.append(-100)
            elif w != prev:
                label_ids.append(tag2id[label[w]])
            else:
                label_ids.append(-100)
            prev = w
        encoded_labels.append(label_ids)

    encodings["labels"] = torch.tensor(encoded_labels)
    return encodings
