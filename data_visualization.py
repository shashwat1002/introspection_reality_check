"""
Plotting for the collated analysis results.

This is the plotting code that used to live in data_viz.ipynb, packaged so the
full pipeline (run_full_pipeline.sh) can call it non-interactively.

Usage:
    python data_visualization.py --results collated_analysis_results_loose.csv --loose
    python data_visualization.py --results collated_analysis_results.csv \
        --models llama-8 --prompt-files experiment_prompts/3_set.jsonl --style square
"""

import argparse
import os
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless: the pipeline runs this on compute nodes

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, Normalize

# =========================
# Model shorthands
# =========================
model_refs = {
    "qwen-72": "Qwen_Qwen2.5-72B-Instruct",
    "qwen-7": "Qwen_Qwen2.5-7B-Instruct",
    "llama-8": "meta-llama_Llama-3.1-8B-Instruct",
    "llama-70": "meta-llama_Llama-3.1-70B-Instruct",
    "llama-33-70": "meta-llama_Llama-3.3-70B-Instruct",
    "smol-lm": "HuggingFaceTB_SmolLM3-3B",
    "gemma": "google_gemma-3-27b-it",
    "qwen-3-32": "Qwen_Qwen3-32B",
}


def resolve_model(model_ref: str) -> str:
    """Accept either a shorthand ("llama-8") or a full model name."""
    if model_ref in model_refs:
        return model_refs[model_ref]
    return model_ref.replace("/", "_")


# =========================
# Loading / filtering
# =========================
# The notebook drops historical runs that were produced before some prompt-file
# typos were fixed. That only makes sense for the big archival CSV, so it is
# opt-in here (--filter-legacy) and off for fresh pipeline runs.
_LEGACY_VECTOR_DIRS = r"outputs/2026-04-2[78]|Archive \(1\)/2026-04-2[789]|outputs/2026-08-01"
_LEGACY_TEXT_MIN_DATE = "2026-05-12"


def filter_legacy_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the vector-steering / text-steering runs the notebook trusts."""
    mask = (df["experiment_type"] != "vector_steering") | (
        df["file"].str.contains(_LEGACY_VECTOR_DIRS, regex=True)
    )
    df = df[mask]

    mask = (df["experiment_type"] != "text_steering") | (
        df["file"].str.extract(r"(\d{4}-\d{2}-\d{2})")[0] >= _LEGACY_TEXT_MIN_DATE
    )
    return df[mask]


def load_results(csv_path: str, filter_legacy: bool = False) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if filter_legacy:
        df = filter_legacy_runs(df)
    df["model"] = df["model"].apply(lambda x: str(x).replace("/", "_"))
    return df


def best_vector_rows(df: pd.DataFrame, randomize_prompt: bool = True) -> pd.DataFrame:
    """Highest vector_injection_mean per (model, prompt_file, randomize_prompt)."""
    dt_t = df[df["randomize_prompt"] == randomize_prompt]
    dt_t = dt_t[dt_t["experiment_type"] == "vector_steering"]
    if dt_t.empty:
        return dt_t
    idx = dt_t.groupby(["model", "prompt_file", "randomize_prompt"])[
        "vector_injection_mean"
    ].idxmax()
    return dt_t.loc[idx]


# =========================
# Stats extraction
# =========================
_STAT_KEYS = [
    "prompt_manipulation_mean",
    "vector_injection_mean",
    "control_mean",
    "invalid_mean",
    "prompt_manipulation_std",
    "vector_injection_std",
    "control_std",
    "invalid_std",
]


def view_stats_map(
    df: pd.DataFrame,
    model_ref: str,
    prompt_file: str,
    vector_agg: str = "max_correct",
    randomize_prompt: bool = True,
    layer=None,
    alpha=None,
    verbose: bool = False,
) -> Tuple[Dict[str, Dict[str, float]], pd.Series]:
    """Filter the dataframe for a given model and prompt file.

    For each experiment type, get the values for prompt_manipulation_mean,
    vector_injection_mean, control_mean.  For the vector_steering rows, use the
    aggregation specified by vector_agg (default is max_correct).
    """
    model = resolve_model(model_ref)
    df_model = df[(df["model"] == model) & (df["prompt_file"] == prompt_file)]

    # control rows: prefer the matching randomize_prompt value, but fall back to whatever
    # exists (e.g. llama-3.3-70B only has randomize_prompt=False control rows in some datasets)
    control_rows = df_model[
        (df_model["experiment_type"] == "control")
        & (df_model["randomize_prompt"] == randomize_prompt)
    ]
    if len(control_rows) == 0:
        control_rows = df_model[df_model["experiment_type"] == "control"]

    # text_steering and vector_steering are filtered by randomize_prompt
    df_filtered = df_model[df_model["randomize_prompt"] == randomize_prompt]

    # best vector_steering row
    vector_rows = df_filtered[df_filtered["experiment_type"] == "vector_steering"]
    if vector_agg == "max_correct":
        if not vector_rows.empty:
            vector_rows = vector_rows[
                vector_rows["vector_injection_mean"]
                == vector_rows["vector_injection_mean"].max()
            ]
    else:
        raise NotImplementedError(f"vector_agg {vector_agg} not implemented")

    if layer is not None and alpha is not None:
        vector_rows = vector_rows[
            (vector_rows["layer"] == layer) & (vector_rows["alpha"] == alpha)
        ]

    non_vector = df_filtered[df_filtered["experiment_type"] == "text_steering"]
    combined = pd.concat([control_rows, non_vector, vector_rows], axis=0)

    all_stats = {}
    for _, row in combined.iterrows():
        experiment_type = row["experiment_type"]
        if verbose:
            print(experiment_type)
        all_stats[experiment_type] = {
            k: row.get(k, float("nan")) for k in _STAT_KEYS
        }
    return all_stats, combined["file"]


# =========================
# Palettes / shared reshaping
# =========================
# single-hue sequential blue ramp (light -> dark); the lightest step is a faint tint rather
# than the page surface, so a near-zero cell still reads as a cell
_SEQ_BLUE = [
    "#eaf2fd", "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
SEQ_BLUE_CMAP = LinearSegmentedColormap.from_list("seq_blue", _SEQ_BLUE)

# same ramp in orange, for a second sequential context (e.g. side-by-side strict/loose)
_SEQ_ORANGE = [
    "#fdefe4", "#fbdfcd", "#f7c8a9", "#f3b184", "#ee9a60", "#eb6834", "#d95926",
    "#c14b1e", "#a53f18", "#8a3413", "#6e290f",
]
SEQ_ORANGE_CMAP = LinearSegmentedColormap.from_list("seq_orange", _SEQ_ORANGE)

_INK = "#0b0b0b"
_INK_MUTED = "#52514e"
_SURFACE = "#fcfcfb"


def _stats_to_frames(stats_map, prompt_file):
    """Shared reshaping: stats dict -> (means_df, stds_df) with clean, ordered labels."""
    mean_data = {exp: {k: v for k, v in vals.items() if k.endswith('_mean')}
                 for exp, vals in stats_map.items()}
    std_data = {exp: {k: v for k, v in vals.items() if k.endswith('_std')}
                for exp, vals in stats_map.items()}

    means_df = pd.DataFrame(mean_data).T
    stds_df = pd.DataFrame(std_data).T

    col_names = [c.replace('_mean', '').replace('_', ' ').title() for c in means_df.columns]
    col_names = [c.replace("Prompt Manipulation", "Gaslight")
                  .replace("Vector Injection", "Hidden\nIntervention") for c in col_names]
    means_df.columns = col_names
    stds_df.columns = col_names

    col_order = [c for c in ["Control", "Hidden\nIntervention", "Gaslight", "Invalid"]
                 if c in means_df.columns]
    means_df = means_df.reindex(columns=col_order)
    stds_df = stds_df.reindex(columns=col_order)

    row_names = [r.replace('_', ' ').title() for r in means_df.index]
    row_names = [r.replace("Text Steering", "Gaslight")
                  .replace("Vector Steering", "Hidden\nIntervention") for r in row_names]
    means_df.index = row_names
    stds_df.index = row_names

    row_order = [r for r in ["Control", "Hidden\nIntervention", "Gaslight"]
                 if r in means_df.index]
    means_df = means_df.reindex(row_order)
    stds_df = stds_df.reindex(row_order)

    if "2" in prompt_file and "vs" not in prompt_file:
        # the gaslight condition doesn't exist in the 2-option prompts
        drop = [c for c in means_df.columns if "Gaslight" in c]
        means_df = means_df.drop(columns=drop)
        stds_df = stds_df.drop(columns=drop)

    return means_df, stds_df


# =========================
# Plots
# =========================
def plot_stats_map(stats_map, model_name, prompt_file, loose=False,
                   prompt_file_display_clean=False, save_dir="plots", save_suffix=""):
    """Plot the stats map as a heatmap.

    Rows  = actual intervention applied.
    Cols  = what the model claims / how the model responds.
    Cells = mean ± std (%).
    """
    LABEL_FONTSIZE = 16
    TICK_FONTSIZE  = 15
    TITLE_FONTSIZE = 16
    ANNOT_FONTSIZE = 17
    CBAR_FONTSIZE  = 17

    # Split into mean / std DataFrames
    mean_data = {exp: {k: v for k, v in vals.items() if k.endswith('_mean')}
                 for exp, vals in stats_map.items()}
    std_data  = {exp: {k: v for k, v in vals.items() if k.endswith('_std')}
                 for exp, vals in stats_map.items()}

    means_df = pd.DataFrame(mean_data).T
    stds_df  = pd.DataFrame(std_data).T

    # Clean column names: strip suffix, underscores -> spaces, title-case
    col_names = [c.replace('_mean', '').replace('_', ' ').title() for c in means_df.columns]
    # rename "Prompt Manipulation" to "Gaslight" and "Vector Injection" to "Hidden Intervention"
    col_names = [c.replace("Prompt Manipulation", "Gaslight").replace("Vector Injection", "Hidden Intervention") for c in col_names]

    means_df.columns = col_names
    stds_df.columns  = col_names

    # reorder columns to Control, Hidden Intervention, Gaslight (if applicable)
    desired_order = [c for c in ["Control", "Hidden Intervention", "Gaslight", "Invalid"]
                     if c in means_df.columns]
    means_df = means_df.reindex(columns=desired_order)
    stds_df  = stds_df.reindex(columns=desired_order)

    # Clean row names
    row_names = [r.replace('_', ' ').title() for r in means_df.index]
    # rename "Text Steering" to "Gaslight" and "Vector Steering" to "Hidden Intervention"
    row_names = [r.replace("Text Steering", "Gaslight").replace("Vector Steering", "Hidden Intervention") for r in row_names]
    means_df.index = row_names
    stds_df.index  = row_names

    # Re-order rows to Control, Hidden Intervention, Gaslight (if applicable)
    desired_order = [r for r in ["Control", "Hidden Intervention", "Gaslight"]
                     if r in means_df.index]
    means_df = means_df.reindex(desired_order)
    stds_df  = stds_df.reindex(desired_order)

    if "2" in prompt_file and "vs" not in prompt_file:
        # remove the "Prompt Manipulation" column for 2-set prompts, since it's not applicable
        means_df = means_df.drop(columns=[col for col in means_df.columns if "Gaslight" in col])
        stds_df  = stds_df.drop(columns=[col for col in stds_df.columns if "Gaslight" in col])

    # Build per-cell annotation strings  "mean\n±std"
    annot = np.empty(means_df.shape, dtype=object)
    for i in range(means_df.shape[0]):
        for j in range(means_df.shape[1]):
            m = means_df.iloc[i, j]
            s = stds_df.iloc[i, j]
            annot[i, j] = f"{m:.1f}\n±{s:.1f}"

    fig, ax = plt.subplots(figsize=(9, max(3, len(stats_map) * 1.4)))

    sns.heatmap(
        means_df, annot=annot, fmt='', cmap='YlOrRd', ax=ax,
        linewidths=0.5, linecolor='lightgray', vmin=0, vmax=100,
        annot_kws={"size": ANNOT_FONTSIZE},
    )

    # Axis labels explaining what rows/cols represent
    ax.set_ylabel("Actual Intervention", fontsize=LABEL_FONTSIZE, labelpad=10, fontweight='bold')
    ax.set_xlabel("Model's Claim", fontsize=LABEL_FONTSIZE, labelpad=10, fontweight='bold')

    # Tick label font sizes
    ax.tick_params(axis='x', labelsize=TICK_FONTSIZE)
    ax.tick_params(axis='y', labelsize=TICK_FONTSIZE)

    # Colorbar (legend) font size
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=CBAR_FONTSIZE)
    cbar.set_label("% responses", fontsize=CBAR_FONTSIZE)

    # Nicer title
    model_display = (
        model_name
        .replace('meta-llama_', '')
        .replace('Qwen_', '')
        .replace('google_', '')
        .replace('HuggingFaceTB_', '')
        .replace('-Instruct', '')
        .replace('_', ' ')
    )
    if not prompt_file_display_clean:
        prompt_display = (
            prompt_file.split('/')[-1]
            .replace('.jsonl', '')
            .replace('_', ' ')
            .title()
        )
    else:
        if "2" in prompt_file:
            prompt_display = "Model has 2 options"
        elif "3" in prompt_file:
            prompt_display = "Model has 3 options"
        else:
            prompt_display = prompt_file.split('/')[-1].replace('.jsonl', '').replace('_', ' ').title()

    ax.set_title(
        f"{model_display}  |  Prompt: {prompt_display}",
        fontsize=TITLE_FONTSIZE, fontweight='bold', pad=14,
    )

    plt.xticks(rotation=30, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    model_name_sanitized  = model_name.replace('/', '_')
    prompt_file_sanitized = prompt_file.split('/')[-1]
    suffix = f"{save_suffix}-loose.png" if loose else f"{save_suffix}.png"
    out_path = f"{save_dir}/{model_name_sanitized}-{prompt_file_sanitized}{suffix}"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved {out_path}")
    return out_path


def plot_stats_map_square(
    stats_map,
    model_name,
    prompt_file,
    loose=False,
    prompt_file_display_clean=False,
    cmap=SEQ_BLUE_CMAP,
    cell_size=1.15,
    show_std=True,
    save_dir="plots-square",
    save_suffix="",
    ax=None,
):
    """Square-celled heatmap of the stats map, on a single-hue sequential ramp.

    Rows  = actual intervention applied.
    Cols  = what the model claims.
    Cells = mean (%) with an optional smaller ±std underneath.

    Same inputs as `plot_stats_map`, but cells are exactly square, the ramp is one
    hue light->dark (magnitude, not rainbow), annotations pick their ink for
    contrast per cell, and grid/axes are recessive. Pass `ax` to draw into an
    existing axes (nothing is saved in that case).
    """
    LABEL_FONTSIZE = 14
    TICK_FONTSIZE = 12
    TITLE_FONTSIZE = 14
    MEAN_FONTSIZE = 15
    STD_FONTSIZE = 10
    CBAR_FONTSIZE = 12

    means_df, stds_df = _stats_to_frames(stats_map, prompt_file)
    n_rows, n_cols = means_df.shape
    values = means_df.to_numpy(dtype=float)

    standalone = ax is None
    if standalone:
        # width leaves room for the row labels + colorbar, height for title + column labels
        fig, ax = plt.subplots(
            figsize=(n_cols * cell_size + 3.2, n_rows * cell_size + 1.6),
            facecolor=_SURFACE,
        )
    else:
        fig = ax.figure
    ax.set_facecolor(_SURFACE)

    norm = Normalize(vmin=0, vmax=100)
    mesh = ax.pcolormesh(
        np.arange(n_cols + 1), np.arange(n_rows + 1), values,
        cmap=cmap, norm=norm, edgecolors=_SURFACE, linewidth=2.0,
    )
    ax.set_aspect("equal")          # square cells
    ax.invert_yaxis()               # first row on top, like the dataframe

    # per-cell annotation, ink chosen against the cell fill
    for i in range(n_rows):
        for j in range(n_cols):
            m = values[i, j]
            if np.isnan(m):
                continue
            r, g, b, _ = cmap(norm(m))
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            mean_ink = "#ffffff" if lum < 0.55 else _INK
            std_ink = "#e8eef7" if lum < 0.55 else _INK_MUTED
            y = i + (0.43 if show_std else 0.5)
            ax.text(j + 0.5, y, f"{m:.1f}", ha="center", va="center",
                    fontsize=MEAN_FONTSIZE, fontweight="semibold", color=mean_ink)
            if show_std:
                s = stds_df.iloc[i, j]
                ax.text(j + 0.5, i + 0.70, f"±{s:.1f}", ha="center", va="center",
                        fontsize=STD_FONTSIZE, color=std_ink)

    # two-word labels wrap, so they stay inside their column
    wrapped = [c.replace(' ', '\n', 1) if len(c) > 10 else c for c in means_df.columns]
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels(wrapped, fontsize=TICK_FONTSIZE, color=_INK)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(means_df.index, fontsize=TICK_FONTSIZE, color=_INK, rotation=0)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.tick_params(length=0, pad=6)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xlabel("Model's Claim", fontsize=LABEL_FONTSIZE, labelpad=10,
                  fontweight="bold", color=_INK)
    ax.set_ylabel("Actual Intervention", fontsize=LABEL_FONTSIZE, labelpad=10,
                  fontweight="bold", color=_INK)

    # slim colorbar, exactly as tall as the grid (axes fraction == data box under equal aspect)
    cax = ax.inset_axes([1.04, 0.0, 0.035 * n_cols / n_rows, 1.0])
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=CBAR_FONTSIZE, length=0, color=_INK_MUTED,
                        labelcolor=_INK_MUTED)
    cbar.set_label("% responses", fontsize=CBAR_FONTSIZE, color=_INK_MUTED, labelpad=8)
    cbar.set_ticks([0, 25, 50, 75, 100])

    model_display = (
        model_name
        .replace('meta-llama_', '')
        .replace('Qwen_', '')
        .replace('google_', '')
        .replace('HuggingFaceTB_', '')
        .replace('-Instruct', '')
        .replace('_', ' ')
    )
    stem = prompt_file.split('/')[-1].replace('.jsonl', '')
    if prompt_file_display_clean and ("2" in stem or "3" in stem):
        prompt_display = f"Model has {'2' if '2' in stem else '3'} options"
    else:
        # e.g. neutral_steer_prompt, which has no option count in its name
        prompt_display = stem.replace('_', ' ').title()

    ax.set_title(f"{model_display}   ·   {prompt_display}",
                 fontsize=TITLE_FONTSIZE, fontweight="bold", pad=24, color=_INK)

    if not standalone:
        return ax

    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    model_name_sanitized = model_name.replace('/', '_')
    prompt_file_sanitized = prompt_file.split('/')[-1]
    suffix = f"{save_suffix}-loose.png" if loose else f"{save_suffix}.png"
    out_path = f"{save_dir}/{model_name_sanitized}-{prompt_file_sanitized}{suffix}"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=_SURFACE)
    print(f"Saved {out_path}")
    return out_path


# =========================
# Driver
# =========================
def plot_all(
    df: pd.DataFrame,
    models: Optional[List[str]] = None,
    prompt_files: Optional[List[str]] = None,
    style: str = "both",
    loose: bool = False,
    randomize_prompt: bool = True,
    plots_dir: str = "plots",
    square_dir: str = "plots-square",
    save_suffix: str = "",
    prompt_file_display_clean: bool = True,
) -> List[str]:
    """Make every (model, prompt_file) plot present in the collated results."""
    models = models or sorted(df["model"].dropna().unique())
    prompt_files = prompt_files or sorted(df["prompt_file"].dropna().unique())

    written = []
    for model_ref in models:
        model = resolve_model(model_ref)
        for prompt_file in prompt_files:
            stats, _dirs = view_stats_map(
                df, model, prompt_file, randomize_prompt=randomize_prompt
            )
            if not stats:
                print(f"no rows for {model} / {prompt_file}, skipping")
                continue
            try:
                if style in ("heatmap", "both"):
                    written.append(plot_stats_map(
                        stats, model, prompt_file, loose=loose,
                        prompt_file_display_clean=prompt_file_display_clean,
                        save_dir=plots_dir, save_suffix=save_suffix,
                    ))
                if style in ("square", "both"):
                    written.append(plot_stats_map_square(
                        stats, model, prompt_file, loose=loose,
                        prompt_file_display_clean=prompt_file_display_clean,
                        save_dir=square_dir, save_suffix=save_suffix,
                    ))
            except Exception as e:
                print(f"Error plotting stats for {model} / {prompt_file}: {e}")
            finally:
                plt.close("all")
    return written


def main():
    parser = argparse.ArgumentParser(
        description="Plot the collated analysis results produced by analysis_script.py"
    )
    parser.add_argument(
        "--results", required=True,
        help="Collated results CSV (output of analysis_script.py)",
    )
    parser.add_argument(
        "--models", nargs="*", default=None,
        help="Model shorthands (e.g. llama-8) or full model names. Default: all in the CSV",
    )
    parser.add_argument(
        "--prompt-files", nargs="*", default=None,
        help="Prompt files to plot. Default: all in the CSV",
    )
    parser.add_argument(
        "--style", choices=["heatmap", "square", "both"], default="both",
        help="heatmap = plot_stats_map, square = plot_stats_map_square",
    )
    parser.add_argument(
        "--loose", action="store_true",
        help="Tag the output filenames with -loose (the CSV came from --loose_parsing)",
    )
    parser.add_argument(
        "--randomize-prompt", choices=["true", "false"], default="true",
        help="Which randomize_prompt rows to plot",
    )
    parser.add_argument("--plots-dir", default="plots")
    parser.add_argument("--square-dir", default="plots-square")
    parser.add_argument(
        "--save-suffix", default="",
        help="Extra suffix inserted before .png (e.g. -multi-number-check)",
    )
    parser.add_argument(
        "--filter-legacy", action="store_true",
        help="Apply the notebook's historical run filters (only for the archival CSV)",
    )
    parser.add_argument(
        "--best-rows-out", default=None,
        help="Also write the best vector-steering row per (model, prompt_file) here",
    )
    args = parser.parse_args()

    df = load_results(args.results, filter_legacy=args.filter_legacy)
    randomize_prompt = args.randomize_prompt == "true"

    if args.best_rows_out:
        best = best_vector_rows(df, randomize_prompt=randomize_prompt)
        best.to_csv(args.best_rows_out, index=False)
        print(f"Saved {args.best_rows_out} ({len(best)} rows)")

    written = plot_all(
        df,
        models=args.models,
        prompt_files=args.prompt_files,
        style=args.style,
        loose=args.loose,
        randomize_prompt=randomize_prompt,
        plots_dir=args.plots_dir,
        square_dir=args.square_dir,
        save_suffix=args.save_suffix,
    )
    print(f"\nWrote {len(written)} plots.")


if __name__ == "__main__":
    main()
