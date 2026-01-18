from gazetteer import load_gazetteer
from preprocess import extract_hindi_sentences
from conll_builder import write_conll

def build_conll():
    gazetteers = {
        "DISEASE": load_gazetteer("data/raw/Disease_Gazetteer.txt"),
        "SYMPTOM": load_gazetteer("data/raw/Symptom_Gazetteer.txt"),
        "CONSUMABLE": load_gazetteer("data/raw/Consumable_Gazetteer.txt"),
        "PERSON": load_gazetteer("data/raw/Person_Gazetteer.txt"),
    }

    sentences = extract_hindi_sentences("data/raw/Hindi_Health_Data.txt")
    write_conll(sentences, gazetteers, "data/processed/train.conll")

build_conll()