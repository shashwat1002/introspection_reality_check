#!/usr/bin/env bash
# Full experiment + analysis pipeline.
#
#   1. eval.py      -> vector steering (activation injection) runs
#   2. gaslight_experiment.py -> text steering ("gaslight") run + control run
#   3. analysis_script.py     -> collated_analysis_results*.csv over those runs
#   4. data_visualization.py  -> heatmaps in <run dir>/plots and <run dir>/plots-square
#
# Everything for one invocation lands under a single run directory so the
# analysis step can just walk it, and nothing overwrites the repo-root
# collated_analysis_results*.csv that are tracked in git. Each model gets its
# own subdirectory inside the run dir.
#
# Models are selected by short key (see --list-models); each key carries the
# layer sweep and normalize setting used for that model in
# run_multijob_all_models.sh, so --layers is only needed to override them.
#
# Examples:
#   ./run_full_pipeline.sh --models llama-8b
#   ./run_full_pipeline.sh --models "llama-8b qwen3-32b" --alphas "0.25 0.5 1 2"
#   ./run_full_pipeline.sh --models all --prompt-file experiment_prompts/3_set_no_conversation.jsonl
#   ./run_full_pipeline.sh --models llama-70b --layers "8 16 24"   # override the sweep

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# =========================
# Model registry
#   key | HF model id | default layer sweep | default normalize | default steer_policy
# Layer sweeps and normalize flags mirror run_multijob_all_models.sh. Policies
# mirror what each model was actually swept with: Qwen3-32B and gemma-3-27b-it
# ran prefill_and_decode_only, the rest ran prefill_only.
# =========================
MODEL_REGISTRY=(
    "llama-8b|meta-llama/Llama-3.1-8B-Instruct|2 4 8 10 16 20 24 27|true|prefill_only"
    "gemma-27b|google/gemma-3-27b-it|2 4 8 16 32 40 48|false|prefill_and_decode_only"
    "qwen3-32b|Qwen/Qwen3-32B|2 4 8 16 24 31|false|prefill_and_decode_only"
    "llama-70b|meta-llama/Llama-3.1-70B-Instruct|2 4 5 8 16 24 32 48 64 79|true|prefill_only"
    "qwen2.5-72b|Qwen/Qwen2.5-72B-Instruct|2 4 5 8 16 24 32 48 64 79|true|prefill_only"
)

# Populates REG_KEY / REG_ID / REG_LAYERS / REG_NORMALIZE / REG_POLICY for a key or full HF id.
registry_lookup() {
    local want="$1" entry key id layers norm policy
    for entry in "${MODEL_REGISTRY[@]}"; do
        IFS='|' read -r key id layers norm policy <<<"$entry"
        if [[ "$want" == "$key" || "$want" == "$id" ]]; then
            REG_KEY="$key"; REG_ID="$id"; REG_LAYERS="$layers"
            REG_NORMALIZE="$norm"; REG_POLICY="$policy"
            return 0
        fi
    done
    return 1
}

list_models() {
    local entry key id layers norm policy
    printf '%-14s %-38s %-28s %-10s %s\n' "KEY" "MODEL" "LAYERS" "NORMALIZE" "STEER_POLICY"
    for entry in "${MODEL_REGISTRY[@]}"; do
        IFS='|' read -r key id layers norm policy <<<"$entry"
        printf '%-14s %-38s %-28s %-10s %s\n' "$key" "$id" "$layers" "$norm" "$policy"
    done
}

# =========================
# Defaults
# =========================
MODELS_ARG="qwen3-32b"  # space separated keys / HF ids, or "all"
PROMPT_FILE="experiment_prompts/2_set_no_conversation.jsonl"
LAYERS=""              # empty -> per-model sweep from the registry
ALPHAS="4"
STEER_POLICY=""        # empty -> per-model value from the registry
NORMALIZE=""           # empty -> per-model value from the registry
RANDOMIZE="true"
N_TRIALS=1000          # eval_efficient.py: total samples per concept (split across permutations)
NUM_SAMPLES=500        # gaslight_experiment.py: samples per noun. analysis_script.py asserts 500/501
BATCH_SIZE=32
RUN_ROOT="pipeline_runs"
RUN_NAME=""
CONDA_ENV="${CONDA_ENV:-multi_312}"
ANALYSIS_MODES="strict loose loose_multi"
PLOT_STYLE="both"
DRY_RUN=0
SKIP_EVAL=0
SKIP_GASLIGHT=0
SKIP_CONTROL=0
SKIP_ANALYSIS=0
SKIP_PLOTS=0

usage() {
    sed -n '2,26p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'USAGE'

Options:
  --models "K1 K2 ..."    models to run: registry keys, HF ids, or "all"
                          (default: llama-8b)
  --model NAME            single model (alias for --models NAME)
  --list-models           print the model registry and exit
  --prompt-file PATH      experiment prompt file (default: experiment_prompts/3_set.jsonl)
  --layers "L1 L2 ..."    layers to inject at, overrides the per-model sweep
  --alphas "A1 A2 ..."    injection strengths    (default: 4)
  --steer-policy P        prefill_only|decode_only|always|prefill_and_decode_only,
                          overrides the per-model value
  --normalize BOOL        normalize the steering vector, overrides the per-model value
  --randomize BOOL        randomize option order in the prompt (default: true)
  --n-trials N            eval_efficient samples per concept (default: 1000)
  --num-samples N         gaslight samples per noun (default: 500; analysis asserts 500/501)
  --batch-size N          eval_efficient generation batch size (default: 32)
  --run-root DIR          parent dir for run dirs (default: pipeline_runs)
  --run-name NAME         run dir name (default: <model(s)>-<prompt>-<timestamp>)
  --analysis-modes "..."  subset of "strict loose loose_multi"
  --plot-style S          heatmap|square|both    (default: both)
  --conda-env NAME        conda env to activate  (default: $CONDA_ENV or multi_312)
  --skip-eval | --skip-gaslight | --skip-control | --skip-analysis | --skip-plots
  --dry-run               print the commands without running them
  -h, --help
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --models)          MODELS_ARG="$2"; shift 2 ;;
        --model)           MODELS_ARG="$2"; shift 2 ;;
        --list-models)     list_models; exit 0 ;;
        --prompt-file)     PROMPT_FILE="$2"; shift 2 ;;
        --layers)          LAYERS="$2"; shift 2 ;;
        --alphas)          ALPHAS="$2"; shift 2 ;;
        --steer-policy)    STEER_POLICY="$2"; shift 2 ;;
        --normalize)       NORMALIZE="$2"; shift 2 ;;
        --randomize)       RANDOMIZE="$2"; shift 2 ;;
        --n-trials)        N_TRIALS="$2"; shift 2 ;;
        --num-samples)     NUM_SAMPLES="$2"; shift 2 ;;
        --batch-size)      BATCH_SIZE="$2"; shift 2 ;;
        --run-root)        RUN_ROOT="$2"; shift 2 ;;
        --run-name)        RUN_NAME="$2"; shift 2 ;;
        --analysis-modes)  ANALYSIS_MODES="$2"; shift 2 ;;
        --plot-style)      PLOT_STYLE="$2"; shift 2 ;;
        --conda-env)       CONDA_ENV="$2"; shift 2 ;;
        --skip-eval)       SKIP_EVAL=1; shift ;;
        --skip-gaslight)   SKIP_GASLIGHT=1; shift ;;
        --skip-control)    SKIP_CONTROL=1; shift ;;
        --skip-analysis)   SKIP_ANALYSIS=1; shift ;;
        --skip-plots)      SKIP_PLOTS=1; shift ;;
        --dry-run)         DRY_RUN=1; shift ;;
        -h|--help)         usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

# =========================
# Resolve the selected models
# =========================
declare -a MODEL_IDS=()
declare -a MODEL_KEYS=()
declare -a MODEL_LAYERS=()
declare -a MODEL_NORMALIZE=()
declare -a MODEL_POLICY=()

if [[ "$MODELS_ARG" == "all" ]]; then
    MODELS_ARG=""
    for entry in "${MODEL_REGISTRY[@]}"; do
        MODELS_ARG+="${entry%%|*} "
    done
fi

for want in $MODELS_ARG; do
    if registry_lookup "$want"; then
        MODEL_KEYS+=("$REG_KEY")
        MODEL_IDS+=("$REG_ID")
        MODEL_LAYERS+=("${LAYERS:-$REG_LAYERS}")
        MODEL_NORMALIZE+=("${NORMALIZE:-$REG_NORMALIZE}")
        MODEL_POLICY+=("${STEER_POLICY:-$REG_POLICY}")
    else
        # Unregistered model: usable, but it has no layer sweep of its own.
        if [[ -z "$LAYERS" ]]; then
            echo "WARNING: '$want' is not in the model registry and --layers was not given;" \
                 "falling back to layer 16. Run --list-models to see the known models." >&2
        fi
        MODEL_KEYS+=("${want//\//_}")
        MODEL_IDS+=("$want")
        MODEL_LAYERS+=("${LAYERS:-16}")
        MODEL_NORMALIZE+=("${NORMALIZE:-true}")
        MODEL_POLICY+=("${STEER_POLICY:-prefill_and_decode_only}")
    fi
done

if [[ "${#MODEL_IDS[@]}" -eq 0 ]]; then
    echo "no models selected (--models was empty)" >&2
    exit 1
fi

# =========================
# Setup
# =========================

if [[ -n "${CONDA_ENV}" && -f ~/miniconda3/etc/profile.d/conda.sh ]]; then
    # shellcheck disable=SC1090
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate "$CONDA_ENV"
fi

if [[ ! -f "$PROMPT_FILE" ]]; then
    echo "prompt file not found: $PROMPT_FILE" >&2
    exit 1
fi

if [[ "$NUM_SAMPLES" != "500" ]]; then
    echo "WARNING: analysis_script.calculate_overall_stats asserts 500 or 501 responses" \
         "per noun; --num-samples $NUM_SAMPLES will make the text-steering analysis fail." >&2
fi

declare -a SANITIZED_MODELS=()
for model in "${MODEL_IDS[@]}"; do
    SANITIZED_MODELS+=("${model//\//_}")
done

PROMPT_STEM=$(basename "$PROMPT_FILE"); PROMPT_STEM=${PROMPT_STEM%.*}
if [[ -z "$RUN_NAME" ]]; then
    if [[ "${#MODEL_IDS[@]}" -eq 1 ]]; then
        RUN_NAME="${SANITIZED_MODELS[0]}-${PROMPT_STEM}-$(date +%Y-%m-%d_%H-%M-%S)"
    else
        RUN_NAME="${#MODEL_IDS[@]}models-${PROMPT_STEM}-$(date +%Y-%m-%d_%H-%M-%S)"
    fi
fi

RUN_DIR="$RUN_ROOT/$RUN_NAME"
PLOTS_DIR="$RUN_DIR/plots"
SQUARE_DIR="$RUN_DIR/plots-square"

run() {
    echo "+ $*"
    if [[ "$DRY_RUN" -eq 0 ]]; then
        "$@"
    fi
}

echo "=============================================="
echo " run dir     : $RUN_DIR"
echo " prompt file : $PROMPT_FILE"
echo " alphas      : $ALPHAS"
echo " models      :"
for i in "${!MODEL_IDS[@]}"; do
    printf '   %-14s %-38s layers=[%s] normalize=%s policy=%s\n' \
        "${MODEL_KEYS[$i]}" "${MODEL_IDS[$i]}" "${MODEL_LAYERS[$i]}" \
        "${MODEL_NORMALIZE[$i]}" "${MODEL_POLICY[$i]}"
done
echo "=============================================="

mkdir -p "$RUN_DIR"

# =========================
# 1 + 2. Per-model runs
# =========================
for i in "${!MODEL_IDS[@]}"; do
    MODEL_NAME="${MODEL_IDS[$i]}"
    MODEL_DIR="$RUN_DIR/${SANITIZED_MODELS[$i]}"
    MODEL_LAYER_SET="${MODEL_LAYERS[$i]}"
    MODEL_NORM="${MODEL_NORMALIZE[$i]}"
    MODEL_STEER_POLICY="${MODEL_POLICY[$i]}"
    VECTOR_DIR="$MODEL_DIR/vector_steering"
    TEXT_DIR="$MODEL_DIR/text_steering"
    CONTROL_DIR="$MODEL_DIR/control"

    echo
    echo "##############################################"
    echo "# model $((i + 1))/${#MODEL_IDS[@]}: $MODEL_NAME"
    echo "##############################################"

    # ---- 1. Vector steering (eval_efficient.py) ----
    if [[ "$SKIP_EVAL" -eq 0 ]]; then
        echo; echo "### [1/4] vector steering -> $VECTOR_DIR"
        for layer in $MODEL_LAYER_SET; do
            for alpha in $ALPHAS; do
                run python eval.py \
                    hydra.run.dir="$VECTOR_DIR" \
                    model.name="$MODEL_NAME" \
                    experiment.prompt_file="$PROMPT_FILE" \
                    experiment.layer_index="$layer" \
                    experiment.alpha="$alpha" \
                    experiment.n_trials="$N_TRIALS" \
                    experiment.batch_size="$BATCH_SIZE" \
                    experiment.steer_policy="$MODEL_STEER_POLICY" \
                    experiment.normalize_steer_vector="$MODEL_NORM" \
                    experiment.randomize_prompt="$RANDOMIZE"
            done
        done
    else
        echo "### [1/4] vector steering: skipped"
    fi

    # ---- 2. Text steering (gaslight_experiment.py), default + control ----
    if [[ "$SKIP_GASLIGHT" -eq 0 ]]; then
        echo; echo "### [2/4] gaslight (default) -> $TEXT_DIR"
        run python gaslight_experiment.py \
            hydra.run.dir="$TEXT_DIR" \
            model.name="$MODEL_NAME" \
            experiment.prompt_file="$PROMPT_FILE" \
            experiment.type=default \
            experiment.num_samples="$NUM_SAMPLES" \
            experiment.randomize_prompt="$RANDOMIZE"
    else
        echo "### [2/4] gaslight (default): skipped"
    fi

    if [[ "$SKIP_CONTROL" -eq 0 ]]; then
        echo; echo "### [2/4] gaslight (control) -> $CONTROL_DIR"
        run python gaslight_experiment.py \
            hydra.run.dir="$CONTROL_DIR" \
            model.name="$MODEL_NAME" \
            experiment.prompt_file="$PROMPT_FILE" \
            experiment.type=control \
            experiment.num_samples="$NUM_SAMPLES" \
            experiment.randomize_prompt="$RANDOMIZE"
    else
        echo "### [2/4] gaslight (control): skipped"
    fi
done

# =========================
# 3. Analysis (analysis_script.py) over the whole run dir
# =========================
declare -a CSVS=()
declare -a CSV_LOOSE=()
declare -a CSV_SUFFIX=()

for mode in $ANALYSIS_MODES; do
    case "$mode" in
        strict)      CSVS+=("$RUN_DIR/collated_analysis_results.csv");                          CSV_LOOSE+=(0); CSV_SUFFIX+=("") ;;
        loose)       CSVS+=("$RUN_DIR/collated_analysis_results_loose.csv");                    CSV_LOOSE+=(1); CSV_SUFFIX+=("") ;;
        loose_multi) CSVS+=("$RUN_DIR/collated_analysis_results_loose_multi_number_check.csv"); CSV_LOOSE+=(1); CSV_SUFFIX+=("-multi-number-check") ;;
        *) echo "unknown analysis mode: $mode" >&2; exit 1 ;;
    esac
done

if [[ "$SKIP_ANALYSIS" -eq 0 ]]; then
    echo; echo "### [3/4] analysis -> $RUN_DIR"
    i=0
    for mode in $ANALYSIS_MODES; do
        out_csv="${CSVS[$i]}"
        case "$mode" in
            strict)      run python analysis_script.py "$RUN_DIR" --output "$out_csv" ;;
            loose)       run python analysis_script.py "$RUN_DIR" --loose_parsing --output "$out_csv" ;;
            loose_multi) run python analysis_script.py "$RUN_DIR" --loose_parsing --multi_number_check --output "$out_csv" ;;
        esac
        i=$((i + 1))
    done
else
    echo "### [3/4] analysis: skipped"
fi

# =========================
# 4. Plots (data_visualization.py)
# =========================
if [[ "$SKIP_PLOTS" -eq 0 ]]; then
    echo; echo "### [4/4] plots -> $PLOTS_DIR, $SQUARE_DIR"
    mkdir -p "$PLOTS_DIR" "$SQUARE_DIR"
    for i in "${!CSVS[@]}"; do
        out_csv="${CSVS[$i]}"
        if [[ "$DRY_RUN" -eq 0 && ! -f "$out_csv" ]]; then
            echo "  no $out_csv, skipping its plots"
            continue
        fi
        loose_flag=()
        [[ "${CSV_LOOSE[$i]}" -eq 1 ]] && loose_flag=(--loose)
        best_rows="${out_csv%.csv}_best_vector_rows.csv"
        run python data_visualization.py \
            --results "$out_csv" \
            --models "${SANITIZED_MODELS[@]}" \
            --prompt-files "$PROMPT_FILE" \
            --style "$PLOT_STYLE" \
            --randomize-prompt "$RANDOMIZE" \
            --plots-dir "$PLOTS_DIR" \
            --square-dir "$SQUARE_DIR" \
            --save-suffix="${CSV_SUFFIX[$i]}" \
            --best-rows-out "$best_rows" \
            "${loose_flag[@]}"
    done
else
    echo "### [4/4] plots: skipped"
fi

echo
echo "=============================================="
echo " done. results in $RUN_DIR"
echo "=============================================="
