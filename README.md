# Hindi Health-Domain NER — MuRIL-based Architectures

Code for *"An Empirical Study of MuRIL-based architectures for Hindi Health Domain NER"*, an empirical comparison of MuRIL-based sequence-labeling architectures for Named Entity Recognition on Hindi healthcare text (**Disease, Symptom, Consumable, Person**), evaluated on the Hindi Health Dataset (HHD).

## Publication

Published in the *2026 IEEE Guwahati Subsection Conference (GCON)*.

**[View on IEEE Xplore →](https://ieeexplore.ieee.org/document/11648992)** (DOI: [10.1109/GCON69192.2026.11648992](https://doi.org/10.1109/GCON69192.2026.11648992))

IEEE Xplore is the official publication record; the PDF in this repo ([`Hindi-Health-NER-Empirical-Study.pdf`](./Hindi-Health-NER-Empirical-Study.pdf)) is the authors' accepted manuscript, included here for convenience. A project page is published via GitHub Pages: **https://coolsheru2526.github.io/HHD-NER/**

Five architectures are implemented and compared:

| # | Architecture | Description |
|---|---|---|
| 1 | **MuRIL + CRF** (Baseline) | Full fine-tuning of MuRIL, sum of last 4 hidden layers, CRF decoding |
| 2 | **Frozen MuRIL + CRF** | MuRIL encoder frozen; only the CRF/emission layer is trained |
| 3 | **MuRIL + Softmax** | Full fine-tuning, token-level softmax classification (no CRF) |
| 4 | **MuRIL + LoRA + CRF** | Parameter-efficient fine-tuning via LoRA adapters (rank 8) on the last 4 layers |
| 5 | **Stacked MuRIL** | MuRIL → BiLSTM → token-level attention → multi-scale CNN (k=3, k=5) → CRF |

## Results (HHD test set, strict IOB2, seqeval)

Mean ± std across 3 independent seeds (42, 123, 2024):

| Model | Precision | Recall | F1-score |
|---|---|---|---|
| MuRIL + CRF (Baseline) | 0.95 ± 0.015 | 0.93 ± 0.004 | 0.94 ± 0.007 |
| MuRIL (Frozen) + CRF | 0.81 ± 0.007 | 0.82 ± 0.004 | 0.82 ± 0.003 |
| MuRIL + Softmax (No CRF) | 0.91 ± 0.018 | 0.94 ± 0.001 | 0.92 ± 0.01 |
| MuRIL + LoRA + CRF | 0.81 ± 0.025 | 0.77 ± 0.014 | 0.79 ± 0.0061 |
| **MuRIL + BiLSTM + Attention + CNN + CRF (Stacked)** | 0.94 ± 0.007 | 0.94 ± 0.085 | **0.94 ± 0.008** |

Category-wise F1:

| Model | Consumable | Disease | Person | Symptom |
|---|---|---|---|---|
| MuRIL + CRF (Baseline) | 0.93 | 0.94 | 0.98 | 0.93 |
| MuRIL (Frozen) + CRF | 0.81 | 0.77 | 0.97 | 0.80 |
| MuRIL + Softmax (No CRF) | 0.91 | 0.92 | 0.97 | 0.91 |
| MuRIL + LoRA + CRF | 0.79 | 0.76 | 0.93 | 0.76 |
| Stacked | 0.93 | 0.96 | 0.98 | 0.94 |

## Repository structure

```
src/
├── Data pipeline
│   ├── preprocess.py        # Extracts clean Hindi sentences from the raw corpus
│   ├── conll_builder.py     # Gazetteer-based BIO tagging -> CoNLL format
│   ├── split_data.py        # 70/15/15 train/dev/test split (fixed seed)
│   └── dataset.py           # MuRIL tokenization + subword label alignment
│
├── Models
│   ├── model_baseline.py    # MuRIL + CRF
│   ├── model_softmax.py     # MuRIL + Softmax (no CRF)
│   ├── model_lora.py        # MuRIL + LoRA + CRF
│   └── model_stacked.py     # MuRIL + BiLSTM + Attention + CNN + CRF
│
├── Training (each trains 3 seeds — 42, 123, 2024 — for robustness stats)
│   ├── train_baseline.py
│   ├── train_frozen.py
│   ├── train_no_crf.py
│   ├── train_lora.py
│   ├── train_stacked.py
│   └── resume_training.py   # Resume the baseline run from a saved checkpoint epoch
│
├── Evaluation
│   ├── evaluation.py        # seqeval strict-IOB2 metrics, BIO validation, entity distribution
│   ├── test_all_models.py   # Evaluates all 5 trained models (all seeds) on the test set
│   ├── compare_results.py   # Aggregates summary_stats.json across experiments into one table
│   └── predict.py           # Evaluate one model on a CoNLL file, or an interactive demo
│
└── preflight_check.py       # Sanity-checks environment, data, and dependencies before training

data/
├── raw/          # Source corpus + per-entity gazetteers
└── processed/    # Generated CoNLL files (train_v2.conll is checked in; splits are regenerated)
```

## Setup

```bash
git clone <your-repo-url>
cd <repo-folder>

python -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows

pip install torch --index-url https://download.pytorch.org/whl/cu118   # pick the right CUDA build for your GPU
pip install -r requirements.txt

# Indic NLP resources (tokenizer/gazetteer support)
git clone https://github.com/anoopkunchukuttan/indic_nlp_resources.git
```

No paths need to be edited — every script resolves paths relative to the repository root.

## Reproducing the results

```bash
# 1. Sanity-check the environment and data files
python src/preflight_check.py

# 2. (Optional) Rebuild the processed CoNLL file from raw data + gazetteers
python src/conll_builder.py

# 3. Create the 70/15/15 train/dev/test split
python src/split_data.py

# 4. Train each architecture (each runs all 3 seeds automatically)
python src/train_baseline.py     # MuRIL + CRF
python src/train_frozen.py       # Frozen encoder + CRF
python src/train_no_crf.py       # MuRIL + Softmax
python src/train_lora.py         # MuRIL + LoRA + CRF
python src/train_stacked.py      # Stacked MuRIL

# 5. Evaluate everything on the held-out test set
python src/test_all_models.py

# 6. Aggregate a side-by-side comparison table
python src/compare_results.py

# Try a single sentence interactively (baseline model)
python src/predict.py demo
```

Each `train_*.py` script writes to its own `weights/<experiment>/seed_<N>/` folder (best checkpoint, training log, and per-run history), plus a `weights/<experiment>/summary_stats.json` with the mean ± std across seeds. `weights/` is git-ignored — trained checkpoints are not committed.

## Training configuration per experiment

| Experiment | Epochs (max) | Early-stop patience | LR |
|---|---|---|---|
| Baseline | 10 | 3 | 2e-5 |
| Frozen encoder | 15 | 3 | 1e-3 (CRF only) |
| Softmax (no CRF) | 5 | 2 | 2e-5 |
| LoRA + CRF | 10 | 3 | 2e-4 |
| Stacked | 15 | 3 | 2e-5 (MuRIL) / 1e-3 (BiLSTM/Attn/CNN/FC) / 5e-4 (CRF) |

All models use AdamW, `ReduceLROnPlateau` on validation F1, and gradient clipping (max norm 1.0). Evaluation uses strict IOB2 entity-level Precision/Recall/F1 via `seqeval`.

## Citation

```bibtex
@INPROCEEDINGS{11648992,
  author={Mishra, Shreyansh and Pandere, Shubham and Sinha, Saugata},
  booktitle={2026 IEEE Guwahati Subsection Conference (GCON)},
  title={An Empirical Study of MuRIL-based architectures for Hindi Health Domain NER},
  year={2026},
  volume={},
  number={},
  pages={1-6},
  keywords={Modeling;Conditional random fields;Labeling;Bidirectional long short term memory;Architecture;Computer architecture;Training;Printing;Sequences;Sequential analysis;Named Entity Recognition;Hindi NER;Healthcare NLP;Sequence labeling;Neural Network;Transformer Fine-Tuning;MuRIL},
  doi={10.1109/GCON69192.2026.11648992}
}
```


## Acknowledgements

- [MuRIL](https://huggingface.co/google/muril-base-cased) (Khanuja et al., 2021)
- [Hindi Health Dataset](https://www.kaggle.com/datasets/aijain/hindi-health-dataset) (A. Jain, Kaggle)
- [Indic NLP Library](https://github.com/anoopkunchukuttan/indic_nlp_resources)
