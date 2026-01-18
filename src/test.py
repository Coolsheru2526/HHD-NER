import torch
from transformers import AutoTokenizer
from model import MuRIL_CRF

# ---------------- CONFIG ----------------
MODEL_PATH = r"D:\ML\NER\hindi\weights\muril_crf_epoch_5.pt"   # saved model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TAGS = ["O", "B-DISEASE", "B-SYMPTOM", "B-CONSUMABLE", "B-PERSON"]
id2tag = {i: t for i, t in enumerate(TAGS)}

tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")

# ---------------------------------------

def predict(sentence: str, model):
    words = sentence.strip().split()

    enc = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True
    )

    enc["input_ids"] = enc["input_ids"].to(DEVICE)
    enc["attention_mask"] = enc["attention_mask"].to(DEVICE)

    with torch.no_grad():
        pred_ids = model(
            enc["input_ids"],
            enc["attention_mask"]
        )[0]  

    mapping = enc.word_ids(batch_index=0)

    results = []
    prev_word = None

    for idx, w_id in enumerate(mapping):
        if w_id is None or w_id == prev_word:
            continue

        tag = id2tag[pred_ids[idx]]
        results.append((words[w_id], tag))
        prev_word = w_id

    return results


def main():
    model = MuRIL_CRF(len(TAGS)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()


    sentence = "तंबाकू छोड़ने के बाद उसकी तबीयत में सुधार हुआ।"


    preds = predict(sentence, model)

    print("\nNER Output:")
    for word, tag in preds:
        print(f"{word:15s} -> {tag}")
    print("-" * 40)


if __name__ == "__main__":
    main()
