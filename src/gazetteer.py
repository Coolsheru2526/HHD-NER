def load_gazetteer(path):
    with open(path, encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())
