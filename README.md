# Paper Information

This is the code release for "Can LLMs Introspect? A Reality Check"

Now published in COLM 2026. 

# Steering Awareness

The root experiment is a reproduction and modification of [Emergent Introspective Awareness in Large Language Models](https://transformer-circuits.pub/2025/introspection/index.html) where the model is asked to report whether a "thought" was injected into its activations. We add a third setting to this experiment where we do a "gaslight" intervention i.e. steering using a text prompt. 
Like in the original experiment all trials are completely independent i.e. the model never sees a trial and a control in the same context. 



Everything is driven by one script, `run_full_pipeline.sh`, which runs four
stages into a single run directory:

1. `eval.py` — vector steering (activation injection) sweeps over layers/alphas
2. `gaslight_experiment.py` — text steering ("gaslight") run plus a control run
3. `analysis_script.py` — collated `collated_analysis_results*.csv`
4. `data_visualization.py` — heatmaps in `<run dir>/plots` and `<run dir>/plots-square`

Results land under `pipeline_runs/<model>-<prompt>-<timestamp>/`, with one
subdirectory per model.

## Setup

The scripts activate a conda env (default `multi_312`); override with
`--conda-env NAME` or the `CONDA_ENV` variable.

## Running the experiments

The experiment condition is chosen by the prompt file.

**2-set** (injected vs. not injected):

```bash
./run_full_pipeline.sh --models qwen3-32b \
    --prompt-file experiment_prompts/2_set_no_conversation.jsonl
```

**3-set** (three trial types):

```bash
./run_full_pipeline.sh --models qwen3-32b \
    --prompt-file experiment_prompts/3_set_no_conversation.jsonl
```

Multiple models, or all of them:

```bash
./run_full_pipeline.sh --models "llama-8b qwen3-32b" --prompt-file experiment_prompts/2_set_no_conversation.jsonl
./run_full_pipeline.sh --models all --prompt-file experiment_prompts/3_set_no_conversation.jsonl
```

Each model key in the registry carries its own layer sweep, normalize flag, and
steering policy; see `./run_full_pipeline.sh --list-models`. Use `--layers`,
`--alphas`, `--steer-policy`, or `--normalize` to override them, `--dry-run` to
print the commands without running anything, and `--skip-eval` /
`--skip-gaslight` / `--skip-control` / `--skip-analysis` / `--skip-plots` to run
a subset of the stages. `./run_full_pipeline.sh --help` lists all options.


## Other scripts

- `steering_effectiveness_check.py` — sanity check that the steering vectors
  actually move the model, using `config_steer/`.
- `analysis_script.py` / `data_visualization.py` can also be pointed at an
  existing run directory to re-analyze or re-plot without re-running the models.

# Neurofeedback experiments

These are experiments built on top of [Language Models Are Capable of Metacognitive Monitoring and Control of Their Internal Activations](https://arxiv.org/abs/2505.13763v1)

Details for the neurofeedback experiments are in the `llm_neurofeedback/`
folder (a submodule) — see `llm_neurofeedback/SCRIPTS.md` for its pipeline and
driver scripts. The `llm_neurofeedback/README.md` is from the original repository from that paper: [https://github.com/sakimarquis/llm_neurofeedback](https://github.com/sakimarquis/llm_neurofeedback)


# BD Experiments 

Experiments based on [Indications of Belief-Guided Agency and Meta-Cognitive Monitoring in Large Language Models](https://arxiv.org/abs/2602.02467) can be found in `bd_experiments_clean`. 
The authors graciously released their experimental data that can be found in the same directory. 

