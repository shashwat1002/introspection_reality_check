"""
Probe accuracy analysis across all llama seed folders.

For each seed folder, trains a logistic regression probe on layer-0 embeddings
(from Llama-3.3-70B-Instruct) to predict:
  - avg_true_dominance_generation_bucket_k3
  - avg_new_dominance_generation_bucket_k3

Using two embedding configurations:
  - subject + target_new
  - target_true + target_new

Training always oversamples to balance classes. Reports mean ± std of test
accuracy (unbalanced and balanced) across multiple random train/test splits.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED_DIR = Path("icl_data_to_share_fk/gemma/46_60")
CSV_FILENAME = "k_mean_buckets/bucket_assignments/all_settings_bucket_assignments_k3.csv"
# MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"  # Use the same model as the original bucket stats for consistency
MODEL_NAME = "google/gemma-3-27b-it"  # Use the base Gemma-3-27b-it for a more direct probe of the raw embeddings, without instruction-tuning effects
DEFAULT_OUTPUT_FILE = "gemma_probe_accuracy_result.json"
DEFAULT_REPORT_FILE = "gemma_probe_accuracy_report.txt"

TARGET_COLUMNS = [
    "avg_true_dominance_generation_bucket_k3",
    "avg_new_dominance_generation_bucket_k3",
]

SPLIT_SEEDS = [4, 5, 21, 42, 67, 12, 19, 16, 27, 31, 37, 45, 52, 88, 99]


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def get_embed_layer(model: AutoModelForCausalLM):
    """Return the token embedding layer for Llama, Gemma, and similar architectures."""
    # Standard path: Llama, Gemma, Gemma2, Gemma3TextModel
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens
    # Gemma3ForConditionalGeneration: model.model is Gemma3Model, which holds
    # the text model under .language_model (a Gemma3TextModel with .embed_tokens)
    if (hasattr(model, "model") and hasattr(model.model, "language_model")
            and hasattr(model.model.language_model, "embed_tokens")):
        return model.model.language_model.embed_tokens
    if hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
        return model.transformer.wte
    raise ValueError(f"Cannot locate embedding layer for model type: {type(model).__name__}")




def extract_layer0_embeddings(
    sentences: list[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    batch_size: int = 64,
) -> torch.Tensor:
    """Mean-pool the token embedding table (layer 0) for each sentence.

    Directly calls the embedding layer — no transformer layers run, so it is fast.
    Applies model-specific post-embedding scaling (e.g. Gemma's sqrt(hidden_size)).
    Returns a (N, hidden_dim) float32 tensor on CPU.
    """
    embed = get_embed_layer(model)
    embed.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_embs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(sentences), batch_size), desc="Embedding", leave=False):
            batch = sentences[i : i + batch_size]
            inputs = tokenizer(
                batch, padding=True, truncation=True, return_tensors="pt"
            ).to(embed.weight.device)
            hidden = embed(inputs["input_ids"]).float()  # (B, T, D)
            mask = inputs["attention_mask"]              # (B, T)
            lengths = mask.sum(dim=1, keepdim=True).float()
            pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / lengths  # (B, D)
            all_embs.append(pooled.cpu())
    return torch.cat(all_embs, dim=0)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def train_probe(
    X: np.ndarray,
    y: np.ndarray,
    split_random_state: int,
    test_split: float = 0.5,
) -> dict:
    """Train a logistic regression probe with oversampled balanced training set.

    Returns test_accuracy and balanced_test_accuracy.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split, random_state=split_random_state
    )

    # Oversample training set so all classes are equally represented
    rng = np.random.default_rng(split_random_state)
    classes, counts = np.unique(y_train, return_counts=True)
    target_count = counts.max()
    keep_idx = np.concatenate(
        [
            rng.choice(np.where(y_train == c)[0], size=target_count, replace=True)
            for c in classes
        ]
    )
    X_train_bal, y_train_bal = X_train[keep_idx], y_train[keep_idx]

    clf = LogisticRegression(max_iter=2000, C=0.5)
    clf.fit(X_train_bal, y_train_bal)

    test_acc = clf.score(X_test, y_test)
    balanced_test_acc = balanced_accuracy_score(y_test, clf.predict(X_test))

    return {"test_accuracy": test_acc, "balanced_test_accuracy": balanced_test_acc}


def evaluate_config(
    X: np.ndarray,
    y: np.ndarray,
    split_seeds: list[int],
) -> dict:
    """Run probe across all split seeds and return mean ± std for both metrics."""
    test_accs, bal_accs = [], []
    for seed in split_seeds:
        res = train_probe(X, y, split_random_state=seed)
        test_accs.append(res["test_accuracy"])
        bal_accs.append(res["balanced_test_accuracy"])

    return {
        "test_accuracy_mean": float(np.mean(test_accs)),
        "test_accuracy_std": float(np.std(test_accs)),
        "balanced_test_accuracy_mean": float(np.mean(bal_accs)),
        "balanced_test_accuracy_std": float(np.std(bal_accs)),
        "test_accuracy_values": [float(v) for v in test_accs],
        "balanced_test_accuracy_values": [float(v) for v in bal_accs],
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(all_results: dict, args) -> str:
    """Return a human-readable plain-text report using tabulate for tables."""
    import datetime
    from tabulate import tabulate

    emb_names = ["subject+target_new", "target_true+target_new"]
    seed_names = list(all_results.keys())

    def fmt(mean, std):
        return f"{mean:.4f} ± {std:.4f}"

    out = []
    out.append("=" * 74)
    out.append("  Probe Accuracy Analysis — Report")
    out.append("=" * 74)
    out.append(f"  Generated : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    out.append(f"  Model     : {args.model_name}")
    out.append(f"  Base dir  : {args.base_dir}")
    out.append(f"  Seeds     : {len(seed_names)}  ({', '.join(seed_names)})")
    out.append(f"  Splits    : {SPLIT_SEEDS}")
    out.append(f"  JSON out  : {args.output}")
    out.append("=" * 74)

    for target_col in TARGET_COLUMNS:
        for emb_name in emb_names:
            rows = []
            all_test_vals, all_bal_vals = [], []
            for seed_name in seed_names:
                m = all_results[seed_name][target_col][emb_name]
                rows.append([
                    seed_name,
                    fmt(m["test_accuracy_mean"], m["test_accuracy_std"]),
                    fmt(m["balanced_test_accuracy_mean"], m["balanced_test_accuracy_std"]),
                ])
                all_test_vals.extend(m["test_accuracy_values"])
                all_bal_vals.extend(m["balanced_test_accuracy_values"])
            rows.append([
                "OVERALL",
                fmt(np.mean(all_test_vals), np.std(all_test_vals)),
                fmt(np.mean(all_bal_vals), np.std(all_bal_vals)),
            ])
            out.append(f"\nTarget    : {target_col}")
            out.append(f"Embeddings: {emb_name}\n")
            out.append(tabulate(
                rows,
                headers=["Seed", "Test Acc (mean ± std)", "Bal. Test Acc (mean ± std)"],
                tablefmt="simple",
            ))

    # Cross-config summary
    summary_rows = []
    for target_col in TARGET_COLUMNS:
        for emb_name in emb_names:
            all_test, all_bal = [], []
            for seed_name in seed_names:
                m = all_results[seed_name][target_col][emb_name]
                all_test.extend(m["test_accuracy_values"])
                all_bal.extend(m["balanced_test_accuracy_values"])
            summary_rows.append([
                target_col, emb_name,
                fmt(np.mean(all_test), np.std(all_test)),
                fmt(np.mean(all_bal), np.std(all_bal)),
            ])
    out.append("\n" + "=" * 74)
    out.append("  Summary — overall averages across all seeds")
    out.append("=" * 74 + "\n")
    out.append(tabulate(
        summary_rows,
        headers=["Target", "Embeddings", "Test Acc", "Bal. Test Acc"],
        tablefmt="simple",
    ))
    out.append("=" * 74)

    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Probe accuracy analysis across llama seed folders.")
    parser.add_argument(
        "--base_dir",
        type=str,
        default=str(SEED_DIR),
        help="Path to the directory containing seed_N folders",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=MODEL_NAME,
    )
    parser.add_argument(
        "--report",
        type=str,
        default=DEFAULT_REPORT_FILE,
        help="Output plain-text report file",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    seed_folders = sorted(base_dir.glob("seed_*"))
    if not seed_folders:
        raise FileNotFoundError(f"No seed_* folders found in {base_dir}")

    print(f"Found {len(seed_folders)} seed folders: {[f.name for f in seed_folders]}")
    print(f"Loading model: {args.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.float16, device_map="auto"
    )
    model.eval()

    all_results = {}

    for seed_folder in seed_folders:
        csv_path = seed_folder / CSV_FILENAME
        if not csv_path.exists():
            print(f"  WARNING: CSV not found at {csv_path}, skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"Processing {seed_folder.name}")
        print(f"{'='*60}")

        df = pd.read_csv(csv_path)
        print(f"  Loaded {len(df)} rows")

        # Extract embeddings for each text field
        print("  Extracting subject embeddings...")
        subject_embs = extract_layer0_embeddings(df["subject"].tolist(), model, tokenizer)

        print("  Extracting target_true embeddings...")
        target_true_embs = extract_layer0_embeddings(df["target_true"].tolist(), model, tokenizer)

        print("  Extracting target_new embeddings...")
        target_new_embs = extract_layer0_embeddings(df["target_new"].tolist(), model, tokenizer)

        # Build the two embedding configurations
        emb_configs = {
            "subject+target_new": torch.cat([subject_embs, target_new_embs], dim=1).numpy(),
            "target_true+target_new": torch.cat([target_true_embs, target_new_embs], dim=1).numpy(),
        }

        seed_results = {}

        for target_col in TARGET_COLUMNS:
            y = df[target_col].values
            seed_results[target_col] = {}

            for emb_name, X in emb_configs.items():
                print(f"  [{target_col}] [{emb_name}] Running {len(SPLIT_SEEDS)} split seeds...")
                metrics = evaluate_config(X, y, SPLIT_SEEDS)
                seed_results[target_col][emb_name] = metrics

                print(
                    f"    test acc:          {metrics['test_accuracy_mean']:.4f} ± {metrics['test_accuracy_std']:.4f}"
                )
                print(
                    f"    balanced test acc: {metrics['balanced_test_accuracy_mean']:.4f} ± {metrics['balanced_test_accuracy_std']:.4f}"
                )

        all_results[seed_folder.name] = seed_results

    # Save results
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    report = generate_report(all_results, args)
    report_path = Path(args.report)
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report saved to {report_path}")

    # Print one table per (embedding config, target column) combination
    seed_names = list(all_results.keys())
    col_width = 28

    for target_col in TARGET_COLUMNS:
        for emb_name in ["subject+target_new", "target_true+target_new"]:
            print(f"\n{'='*70}")
            print(f"  Target   : {target_col}")
            print(f"  Embeddings: {emb_name}")
            print(f"{'='*70}")
            print(f"{'Seed':<10} {'TestAcc (mean±std)':<{col_width}} {'BalTestAcc (mean±std)':<{col_width}}")
            print(f"{'-'*10} {'-'*(col_width-1)} {'-'*(col_width-1)}")
            all_test_vals, all_bal_vals = [], []
            for seed_name in seed_names:
                metrics = all_results[seed_name][target_col][emb_name]
                test_str = f"{metrics['test_accuracy_mean']:.4f} ± {metrics['test_accuracy_std']:.4f}"
                bal_str  = f"{metrics['balanced_test_accuracy_mean']:.4f} ± {metrics['balanced_test_accuracy_std']:.4f}"
                print(f"{seed_name:<10} {test_str:<{col_width}} {bal_str:<{col_width}}")
                all_test_vals.extend(metrics['test_accuracy_values'])
                all_bal_vals.extend(metrics['balanced_test_accuracy_values'])

            print(f"{'-'*10} {'-'*(col_width-1)} {'-'*(col_width-1)}")
            avg_test_str = f"{np.mean(all_test_vals):.4f} ± {np.std(all_test_vals):.4f}"
            avg_bal_str  = f"{np.mean(all_bal_vals):.4f} ± {np.std(all_bal_vals):.4f}"
            print(f"{'AVG':<10} {avg_test_str:<{col_width}} {avg_bal_str:<{col_width}}")


if __name__ == "__main__":
    main()
