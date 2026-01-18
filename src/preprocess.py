import re

def clean_line(line):
    # remove English in brackets
    line = re.sub(r'\([^)]*[A-Za-z][^)]*\)', '', line)
    # remove pure English words
    line = re.sub(r'[A-Za-z]+', '', line)
    return line.strip()

def is_hindi_sentence(line, threshold=0.4):
    if not line:
        return False
    hindi_chars = sum(1 for c in line if '\u0900' <= c <= '\u097F')
    return hindi_chars / len(line) >= threshold

def extract_hindi_sentences(path):
    sentences = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = clean_line(line)
            if is_hindi_sentence(line):
                sentences.append(line)
    return sentences
