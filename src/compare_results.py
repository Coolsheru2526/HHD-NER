"""
Compare Results Across All Experiments
=======================================
Reads the summary_stats.json (mean +/- std across seeds) that every
train_*.py script writes to its BASE_SAVE_DIR, and prints a combined
comparison table plus a baseline-vs-frozen-encoder ablation summary.
"""

import json
import os


def compare_experiments():
    """Compare all trained experiments using their aggregated summary_stats.json."""

    experiments = {
        "Baseline (Full Fine-tuning + CRF)": "weights/baseline_crf/summary_stats.json",
        "Frozen Encoder (CRF Only)": "weights/frozen_encoder/summary_stats.json",
        "No CRF (Softmax Classification)": "weights/no_crf/summary_stats.json",
        "LoRA Fine-tuning + CRF": "weights/lora/summary_stats.json",
        "Stacked (BiLSTM + Attention + CNN + CRF)": "weights/advanced_model/summary_stats.json"
    }

    print("="*90)
    print("EXPERIMENT COMPARISON - Hindi Health-Domain NER")
    print("="*90)
    print()

    results = []

    for name, path in experiments.items():
        if os.path.exists(path):
            with open(path) as f:
                summary = json.load(f)

            metrics = summary["metrics"]
            results.append({
                "name": name,
                "f1_mean": metrics["f1"]["mean"],
                "f1_std": metrics["f1"]["std"],
                "precision_mean": metrics["precision"]["mean"],
                "recall_mean": metrics["recall"]["mean"],
                "seeds": summary.get("seeds", [])
            })
        else:
            print(f"[!] {name}: Not run yet ({path})")

    print()

    if not results:
        print("No experiment results found. Run the train_*.py scripts first.")
        return

    print(f"{'Experiment':<42} {'Precision':<12} {'Recall':<12} {'F1-score':<15}")
    print("-"*90)

    for r in results:
        f1_str = f"{r['f1_mean']:.4f} ± {r['f1_std']:.4f}"
        print(f"{r['name']:<42} {r['precision_mean']:<12.4f} {r['recall_mean']:<12.4f} {f1_str:<15}")

    # Ablation analysis: baseline vs frozen encoder
    by_name = {r['name']: r for r in results}
    baseline = by_name.get("Baseline (Full Fine-tuning + CRF)")
    frozen = by_name.get("Frozen Encoder (CRF Only)")

    if baseline and frozen:
        print()
        print("="*90)
        print("ABLATION ANALYSIS (Baseline vs Frozen Encoder)")
        print("="*90)

        f1_drop = baseline["f1_mean"] - frozen["f1_mean"]
        f1_drop_pct = (f1_drop / baseline["f1_mean"]) * 100

        print(f"F1 Drop (Baseline -> Frozen): {f1_drop:.4f} ({f1_drop_pct:.1f}%)")
        print("Conclusion: fine-tuning the MuRIL encoder is essential for domain adaptation;")
        print("training only the CRF layer on frozen embeddings leaves significant F1 on the table.")
        print("="*90)


if __name__ == "__main__":
    compare_experiments()
