"""
Check steering effectiveness: for each steering CSV (from neutral_steer_prompt.jsonl runs),
compute the ratio of completions where the steering concept word appears in the generation.
"""

import os
import re
import csv
import yaml
import argparse
from pathlib import Path
from collections import defaultdict


def parse_filename(filename: str):
    """Extract model, layer, alpha from filenames like $google_gemma-3-27b-it_layer$2_alpha$1.csv"""
    stem = Path(filename).stem  # e.g. $google_gemma-3-27b-it_layer$2_alpha$1
    match = re.match(r"^\$(.+)_layer\$(\d+)_alpha\$([\d.]+)$", stem)
    if not match:
        return None
    model = match.group(1).replace("_", "/", 1)  # restore first _ to / for org/model format
    layer = int(match.group(2))
    alpha = float(match.group(3))
    return model, layer, alpha


def find_config(directory: Path):
    """Find the config yaml file in a run directory."""
    for f in directory.iterdir():
        if f.suffix in (".yaml", ".yml") and f.name.startswith("config_actual"):
            return f
    return None


def check_prompt_file(config_path: Path) -> bool:
    """Return True if the config uses neutral_steer_prompt.jsonl."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    prompt_file = cfg.get("experiment", {}).get("prompt_file", "")
    return "neutral_steer_prompt.jsonl" in prompt_file


def concept_in_generation(concept: str, generation: str) -> bool:
    """Case-insensitive whole-word check for concept in generation."""
    pattern = r"\b" + re.escape(concept.lower()) + r"\b"
    return bool(re.search(pattern, generation.lower()))


def process_csv(csv_path: Path, model: str, layer: int, alpha: float) -> dict:
    """Return per-concept hit ratios for a single CSV file."""
    concept_hits = defaultdict(int)
    concept_total = defaultdict(int)

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            concept = row.get("injected_thought", "").strip()
            generation = row.get("generation", "")
            if not concept:
                continue
            concept_total[concept] += 1
            if concept_in_generation(concept, generation):
                concept_hits[concept] += 1

    results = {}
    for concept, total in concept_total.items():
        hits = concept_hits[concept]
        results[concept] = {"hits": hits, "total": total, "ratio": hits / total if total else 0.0}

    return results


def process_directory(directory: Path):
    config_path = find_config(directory)
    if config_path is None:
        return None, "no config found"
    if not check_prompt_file(config_path):
        return None, "prompt_file is not neutral_steer_prompt.jsonl"

    run_results = []
    for csv_file in directory.glob("*.csv"):
        parsed = parse_filename(csv_file.name)
        if parsed is None:
            continue
        model, layer, alpha = parsed
        concept_ratios = process_csv(csv_file, model, layer, alpha)
        run_results.append({
            "file": csv_file.name,
            "model": model,
            "layer": layer,
            "alpha": alpha,
            "concepts": concept_ratios,
        })

    return run_results, None


def print_results(all_results: list):
    for run in all_results:
        overall_hits = sum(c["hits"] for c in run["concepts"].values())
        overall_total = sum(c["total"] for c in run["concepts"].values())
        overall_ratio = overall_hits / overall_total if overall_total else 0.0

        print(f"\n{'='*60}")
        print(f"File:  {run['file']}")
        print(f"Model: {run['model']}  |  Layer: {run['layer']}  |  Alpha: {run['alpha']}")
        print(f"Overall: {overall_hits}/{overall_total} = {overall_ratio:.3f}")
        print(f"{'─'*40}")
        for concept, stats in sorted(run["concepts"].items()):
            bar = "█" * int(stats["ratio"] * 20)
            print(f"  {concept:<20} {stats['hits']:>4}/{stats['total']:<4}  {stats['ratio']:.3f}  {bar}")


def main():
    parser = argparse.ArgumentParser(description="Check steering concept word appearance in generations.")
    parser.add_argument("directory", help="Root directory to search (can contain subdirectories of runs)")
    parser.add_argument("--recursive", "-r", action="store_true",
                        help="Search subdirectories recursively (each subdir treated as a run directory)")
    parser.add_argument("--csv-out", help="Optional: save summary results to a CSV file")
    args = parser.parse_args()

    root = Path(args.directory)
    if not root.exists():
        print(f"Directory not found: {root}")
        return

    # Collect candidate directories
    if args.recursive:
        dirs = [d for d in root.rglob("*") if d.is_dir()]
        dirs = [root] + dirs
    else:
        dirs = [root]

    all_results = []
    for d in dirs:
        results, err = process_directory(d)
        if err:
            if err != "prompt_file is not neutral_steer_prompt.jsonl":
                # Only show unexpected errors, not the expected filter
                pass
            continue
        if results:
            for r in results:
                r["directory"] = str(d)
            all_results.extend(results)

    if not all_results:
        print("No qualifying runs found.")
        return

    print_results(all_results)

    if args.csv_out:
        out_rows = []
        for run in all_results:
            for concept, stats in run["concepts"].items():
                out_rows.append({
                    "directory": run.get("directory", ""),
                    "file": run["file"],
                    "model": run["model"],
                    "layer": run["layer"],
                    "alpha": run["alpha"],
                    "concept": concept,
                    "hits": stats["hits"],
                    "total": stats["total"],
                    "ratio": stats["ratio"],
                })
        with open(args.csv_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=out_rows[0].keys())
            writer.writeheader()
            writer.writerows(out_rows)
        print(f"\nSaved summary to {args.csv_out}")


if __name__ == "__main__":
    main()
