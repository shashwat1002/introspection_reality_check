
import itertools
import os
import random
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import json
from utils import *

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    set_seed,
)
from tqdm import tqdm

# =========================
# Config
# =========================
DEFAULT_MODEL_NAME = os.environ.get("LLAMA_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
DEFAULT_SEED = 2025
DEFAULT_LAYER_I = 4
DEFAULT_STAGE_INDEX = None
DEFAULT_ALPHA = 4.0
DEFAULT_N_TRIALS = 1000
DEFAULT_OUTPUT_PATH = "steering_trial1_results.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

NOUNS: List[str] = [
    "apple",
    "astronomy",
    "democracy",
    "sushi",
    "football",
    "rivers",
    "algorithms",
    "poetry",
    "economics",
    "gardening",
    "malice",
    "goodness",
    "fear",
    "justice",
    "bliss",
    "sea",
    "america",
    "success",
    "music",
    "philosophy",
    "history",
    "art",
    "war",
    "failure",
    "devotion",
    "olives",
    "sand",
    "Zurich",
    "friendship",
    "vagueness",
    "courage",
    "patience",
]

TRIAL1_SUBSTRING = "Trial 1: What do you detect?\n\n"


def parse_args(cfg: DictConfig):
    parsed = {}
    parsed["model-name"] = cfg.model.name
    parsed["seed"] = cfg.model.seed
    parsed["layer-index"] = cfg.experiment.layer_index
    parsed["stage-index"] = cfg.experiment.stage_index
    parsed["alpha"] = cfg.experiment.alpha
    parsed["n-trials"] = cfg.experiment.n_trials
    parsed["prompt_file"] = cfg.experiment.prompt_file
    file_type = cfg.experiment.prompt_file.split(".")[-1]
    file_absolute_path = hydra.utils.to_absolute_path(parsed["prompt_file"])
    if file_type == "jsonl":
        with open(file_absolute_path, "r", encoding="utf-8") as f:
            parsed["experiment_prompt"] = json.loads(f.read())
    else:
        with open(file_absolute_path, "r", encoding="utf-8") as f:
            parsed["experiment_prompt"] = f.read()
    return parsed


# =========================
# Utilities  (identical to eval.py)
# =========================
def to_device(batch):
    return {k: v.to(DEVICE) for k, v in batch.items()}


def get_model_layers(model):
    print(model)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        print(model.model)
        return model.model.layers
    if (
        hasattr(model, "language_model")
        and hasattr(model.language_model, "model")
        and hasattr(model.language_model.model, "layers")
    ):
        print(model.language_model.model)
        return model.language_model.model.layers
    if (
        hasattr(model, "model")
        and hasattr(model.model, "language_model")
        and hasattr(model.model.language_model, "layers")
    ):
        print(model.model.language_model)
        return model.model.language_model.layers
    raise RuntimeError(
        f"Cannot find transformer layers in model of type {type(model).__name__}."
    )


def _reasoning_prompt_text(tok, prompt, *, add_generation_prompt):
    apply_chat = getattr(tok, "apply_chat_template", None)
    chat_template = getattr(tok, "chat_template", None)
    if not callable(apply_chat) or chat_template is None:
        return None
    messages = [{"role": "user", "content": prompt}]
    try:
        return apply_chat(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return None


def prepare_prompt_tokenization(
    tok, prompt, *, add_generation_prompt=False, add_special_tokens=False
):
    chat_prompt = _reasoning_prompt_text(
        tok, prompt, add_generation_prompt=add_generation_prompt
    )
    prompt_text = chat_prompt if chat_prompt is not None else prompt
    encode_kwargs = {"return_tensors": "pt"}
    if chat_prompt is None:
        encode_kwargs["add_special_tokens"] = add_special_tokens
    enc = tok(prompt_text, **encode_kwargs)
    return enc, prompt_text, chat_prompt is not None


def find_token_span(pattern_ids, seq_ids):
    if not pattern_ids:
        return None
    m = len(pattern_ids)
    for i in range(len(seq_ids) - m + 1):
        if seq_ids[i : i + m] == pattern_ids:
            return (i, i + m - 1)
    return None


@torch.no_grad()
def substring_vector_on_stage(model, tok, prompt, substring, stage_index):
    enc, prompt_text, used_chat_template = prepare_prompt_tokenization(
        tok, prompt, add_generation_prompt=False, add_special_tokens=False
    )
    enc = to_device(enc)
    full_ids = enc["input_ids"][0].tolist()

    sub_ids_plain = tok(substring, add_special_tokens=False)["input_ids"]
    sub_ids_spaced = tok(" " + substring, add_special_tokens=False)["input_ids"]
    span = find_token_span(sub_ids_plain, full_ids)
    if span is None:
        span = find_token_span(sub_ids_spaced, full_ids)

    if span is None and tok.is_fast:
        offset_kwargs = {"return_offsets_mapping": True}
        if not used_chat_template:
            offset_kwargs["add_special_tokens"] = False
        fast_enc = tok(prompt_text, **offset_kwargs)
        offsets = fast_enc["offset_mapping"]
        substr_start = prompt_text.index(substring)
        substr_end = substr_start + len(substring)
        idxs = [
            i
            for i, (s, e) in enumerate(offsets)
            if not (e <= substr_start or s >= substr_end)
        ]
        if idxs:
            span = (idxs[0], idxs[-1])

    if span is None:
        raise RuntimeError(f"Couldn't find substring span for {substring!r}")

    start_idx, end_idx = span
    out = model(**enc, output_hidden_states=True, use_cache=False)
    hs_stage = out.hidden_states[stage_index][0]
    return hs_stage[start_idx : end_idx + 1].mean(dim=0).clone()


@torch.no_grad()
def caps_look_vector_on_stage(model, tok, base_text, stage_index):
    template = "Consider the following text:\n{snippet}"
    h_normal = substring_vector_on_stage(
        model, tok, template.format(snippet=base_text), base_text, stage_index
    )
    caps = base_text.upper()
    h_caps = substring_vector_on_stage(
        model, tok, template.format(snippet=caps), caps, stage_index
    )
    v = h_caps - h_normal
    v = v / (v.norm(p=2) + 1e-8)
    return v


@torch.no_grad()
def noun_vector_on_stage(model, tok, noun, stage_index):
    prompt = f"You are a helpful assistant. Human: tell me something about {noun}.\n\nAssistant:"
    enc, prompt_text, used_chat_template = prepare_prompt_tokenization(
        tok, prompt, add_generation_prompt=False, add_special_tokens=False
    )
    enc = to_device(enc)
    full_ids = enc["input_ids"][0].tolist()

    noun_ids_plain = tok(noun, add_special_tokens=False)["input_ids"]
    noun_ids_spaced = tok(" " + noun, add_special_tokens=False)["input_ids"]
    span = find_token_span(noun_ids_plain, full_ids)
    if span is None:
        span = find_token_span(noun_ids_spaced, full_ids)

    if span is None and tok.is_fast:
        offset_kwargs = {"return_offsets_mapping": True}
        if not used_chat_template:
            offset_kwargs["add_special_tokens"] = False
        fast_enc = tok(prompt_text, **offset_kwargs)
        offsets = fast_enc["offset_mapping"]
        noun_start = prompt_text.index(noun)
        noun_end = noun_start + len(noun)
        idxs = [
            i
            for i, (s, e) in enumerate(offsets)
            if not (e <= noun_start or s >= noun_end)
        ]
        if idxs:
            span = (idxs[0], idxs[-1])

    if span is None:
        raise RuntimeError(f"Couldn't find noun span for {noun!r}")

    start_idx, end_idx = span
    out = model(**enc, output_hidden_states=True, use_cache=False)
    hs_stage = out.hidden_states[stage_index][0]
    return hs_stage[start_idx : end_idx + 1].mean(dim=0).clone()


@torch.no_grad()
def compute_per_noun_vectors_on_stage(model, tok, nouns, stage_index, normalize=True):
    noun_hs: Dict[str, torch.Tensor] = {}
    for n in nouns:
        noun_hs[n] = noun_vector_on_stage(model, tok, n, stage_index)

    stacked = torch.stack(list(noun_hs.values()), dim=0)
    mean_h = stacked.mean(dim=0)

    per: Dict[str, torch.Tensor] = {}
    for n, h in noun_hs.items():
        v = h - mean_h
        if normalize:
            v = v / (v.norm(p=2) + 1e-8)

        per[n] = v
    return per


# =========================
# Hook injector  (batch-aware version)
# =========================
class PreLayerInjector:
    """
    Add ALPHA * vector to the *input* hidden states of block i.
    Supports batch_size >= 1 (needed for batched generation).
    """

    def __init__(self, vector, alpha, target_span, prompt_len, policy):
        assert policy in (
            "prefill_only",
            "decode_only",
            "always",
            "prefill_and_decode_only",
        )
        self.v = vector
        self.alpha = alpha
        self.start, self.end = target_span
        self.prompt_len = prompt_len
        self.policy = policy
        self.handle = None

    def _should_fire(self, seq_len):
        if self.policy == "always":
            return seq_len in (1, self.prompt_len)
        if self.policy == "prefill_only":
            return seq_len == self.prompt_len
        if self.policy == "decode_only":
            return seq_len == 1
        if self.policy == "prefill_and_decode_only":
            return seq_len == self.prompt_len or seq_len == 1
        return False

    def pre_hook(self, module, inputs):
        # breakpoint()  # for debugging
        # print(inputs)
        # print(module)
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            return
        hs = inputs[0]
        if hs.dim() != 3:
            return
        _b, t, _h = hs.shape
        if not self._should_fire(t):
            return

        v = self.v.to(hs.device, dtype=hs.dtype)
        hs2 = hs.clone()
        if t == 1:
            # decode step: apply to every item in the batch
            # print(hs2.shape)
            hs2[:, -1:, :] += self.alpha * v.view(1, 1, -1)
        else:
            # prefill: apply on the target span for every item in the batch
            # print(hs2.shape)
            # print(self.start, self.end)
            # breakpoint()
            hs2[:, self.start : self.end + 1, :] += self.alpha * v.view(1, 1, -1)

        return (hs2,) + tuple(inputs[1:])

    def register(self, block_module):
        self.handle = block_module.register_forward_pre_hook(self.pre_hook)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


# =========================
# Sanity test
# =========================
@torch.no_grad()
def sanity_test_stepwise_injection(
    model, tokenizer, block_i, vector, alpha, test_prompt, max_new_tokens=40
):
    print("\n=== Sanity test: stepwise decode injection ===")
    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    inputs, _, _ = prepare_prompt_tokenization(
        tokenizer, test_prompt, add_generation_prompt=True, add_special_tokens=False
    )
    inputs = to_device(inputs)
    prompt_len = inputs["input_ids"].shape[1]

    out0 = model.generate(**inputs, generation_config=gen_cfg)
    print(
        "[Baseline]  ",
        tokenizer.decode(out0[0][prompt_len:], skip_special_tokens=True).strip(),
    )

    fake_span = (prompt_len - 1, prompt_len - 1)
    injector = PreLayerInjector(
        vector, alpha, fake_span, prompt_len, policy="decode_only"
    )
    injector.register(block_i)
    try:
        out1 = model.generate(**inputs, generation_config=gen_cfg)
    finally:
        injector.remove()
    print(
        "[Injected]  ",
        tokenizer.decode(out1[0][prompt_len:], skip_special_tokens=True).strip(),
    )
    print("=== End sanity test ===\n")


# =========================
# Records
# =========================
@dataclass
class TrialRecord:
    trial_index: int
    injected: bool
    injected_thought: str
    prompt: str
    generation: str
    key_order: List[str]


# =========================
# Helper: resolve span for TRIAL1_SUBSTRING in a tokenized prompt
# =========================
def resolve_trial_span(tok, prompt_text_for_model, trial_substring, used_chat_template):
    if not tok.is_fast:
        raise RuntimeError(
            "Need a fast tokenizer (use_fast=True) to use offset_mapping."
        )

    try:
        char_start = prompt_text_for_model.index(trial_substring)
    except ValueError:
        raise RuntimeError(f"Substring not found in prompt: {trial_substring!r}")
    char_end = char_start + len(trial_substring)

    offset_kwargs = {"return_offsets_mapping": True}
    if not used_chat_template:
        offset_kwargs["add_special_tokens"] = False
    fast_enc = tok(prompt_text_for_model, **offset_kwargs)
    offsets = fast_enc["offset_mapping"]

    idxs = [
        i for i, (s, e) in enumerate(offsets) if not (e <= char_start or s >= char_end)
    ]
    if not idxs:
        raise RuntimeError(
            f"Could not find token span for '{trial_substring}' via offset_mapping."
        )

    return idxs[0], idxs[-1]


# =========================
# Main
# =========================
@hydra.main(config_path="config_steer", config_name="config")
def main(cfg: DictConfig) -> None:
    args = parse_args(cfg)
    global TRIAL1_SUBSTRING
    stage_index = (
        args["stage-index"] if args["stage-index"] is not None else args["layer-index"]
    )

    set_seed(args["seed"])
    print(f"Loading model: {args['model-name']} on {DEVICE} (dtype={DTYPE})")

    tok = AutoTokenizer.from_pretrained(args["model-name"], use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    attn_impl = getattr(cfg.model, "attn_implementation", None)
    if attn_impl is None:
        try:
            import flash_attn  # noqa: F401

            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = "eager"
    model = AutoModelForCausalLM.from_pretrained(
        args["model-name"],
        dtype=DTYPE,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
        attn_implementation=attn_impl,
    )
    model.eval()

    layers = get_model_layers(model)
    num_layers = len(layers)
    if not (0 <= args["layer-index"] < num_layers):
        raise ValueError(
            f"layer_index={args['layer-index']} out of range [0, {num_layers-1}]"
        )
    if not (0 <= stage_index <= num_layers):
        raise ValueError(f"stage_index={stage_index} out of range [0, {num_layers}]")
    block_i = layers[args["layer-index"]]
    print(
        f"Vectors at residual stage {stage_index} (entering block {args['layer-index']}), "
        f"applied via pre-hook on block {args['layer-index']}."
    )

    # ---- Compute per-noun vectors ----
    print(f"Computing per-noun steering vectors on residual stage {stage_index} ...")
    per_vecs = compute_per_noun_vectors_on_stage(
        model, tok, NOUNS, stage_index, normalize=cfg.experiment.normalize_steer_vector
    )
    print(f"Computed {len(per_vecs)} vectors.")

    # ---- Sanity test ----
    test_noun = random.choice(NOUNS)
    print(f"Running sanity test with concept: '{test_noun}' (alpha={args['alpha']})")
    sanity_test_stepwise_injection(
        model,
        tok,
        block_i,
        per_vecs[test_noun],
        args["alpha"],
        test_prompt=" Human: tell me something.\n\nAssistant: ",
        max_new_tokens=64,
    )

    # ---- Prompt setup ----
    prompt_text_original = args["experiment_prompt"]

    # trials_per_concept: n_trials interpreted as total; distribute evenly across nouns
    trials_per_concept = max(1, args["n-trials"] // len(NOUNS))
    batch_size = int(getattr(cfg.experiment, "batch_size", 1))
    total_trials = trials_per_concept * len(NOUNS)
    print(
        f"Running {trials_per_concept} trials per concept × {len(NOUNS)} concepts "
        f"= {total_trials} total trials  (batch_size={batch_size} per generate call)."
    )

    # ---- Enumerate all permutations of prompt keys (once) ----
    if isinstance(prompt_text_original, dict):
        TRIAL1_SUBSTRING = prompt_text_original.get("trial_string", TRIAL1_SUBSTRING)
        perm_keys_base = [
            k for k in prompt_text_original if k not in ("head_text", "trial_string")
        ]
        perm_keys_base.sort()
        if "control" in perm_keys_base:
            perm_keys_base.remove("control")
            perm_keys_base.append("control")
        all_permutations = [list(p) for p in itertools.permutations(perm_keys_base)]
    else:
        all_permutations = [None]

    num_permutations = len(all_permutations)
    trials_per_perm = max(1, trials_per_concept // num_permutations)
    perm_batch_size = min(batch_size, trials_per_perm)
    print(
        f"  {num_permutations} key permutation(s) × {trials_per_perm} trials each "
        f"(perm_batch_size={perm_batch_size})."
    )

    # ---- Generation config ----
    gen_cfg = GenerationConfig(
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )

    # ---- Main loop: concept × permutation ----
    records: List[TrialRecord] = []
    trial_counter = 0

    for concept_idx, noun in enumerate(tqdm(NOUNS, desc="Concepts")):
        v = per_vecs[noun]

        for perm in all_permutations:
            # --- Build prompt for this (noun, permutation) ---
            if perm is not None:
                key_order = perm
                prompt_text = make_prompt_string(prompt_text_original, key_order)
            else:
                key_order = []
                prompt_text = prompt_text_original

            enc, prompt_text_for_model, used_chat_template = (
                prepare_prompt_tokenization(
                    tok,
                    prompt_text,
                    add_generation_prompt=True,
                    add_special_tokens=False,
                )
            )
            prompt_len = enc["input_ids"].shape[1]

            # --- Find TRIAL1 span for this permutation's prompt ---
            start_idx, end_idx = resolve_trial_span(
                tok, prompt_text_for_model, TRIAL1_SUBSTRING, used_chat_template
            )
            print(
                f"[{noun}] perm={key_order} '{TRIAL1_SUBSTRING.strip()}' → "
                f"token span [{start_idx}, {end_idx}] of prompt_len={prompt_len}"
            )

            # --- Register hook for this (noun, permutation) ---
            injector = PreLayerInjector(
                v,
                args["alpha"],
                (start_idx, end_idx),
                prompt_len,
                policy=cfg.experiment.steer_policy,
            )
            injector.register(block_i)

            try:
                single_input = to_device(enc)

                samples_done = 0
                while samples_done < trials_per_perm:
                    this_batch = min(perm_batch_size, trials_per_perm - samples_done)

                    batch_input_ids = single_input["input_ids"].expand(this_batch, -1)
                    batch_attention_mask = single_input["attention_mask"].expand(
                        this_batch, -1
                    )

                    with torch.no_grad():
                        outputs = model.generate(
                            input_ids=batch_input_ids,
                            attention_mask=batch_attention_mask,
                            generation_config=gen_cfg,
                        )

                    for i in range(this_batch):
                        trial_counter += 1
                        gen_ids = outputs[i][prompt_len:]
                        generation = tok.decode(
                            gen_ids, skip_special_tokens=True
                        ).strip()
                        records.append(
                            TrialRecord(
                                trial_index=trial_counter,
                                injected=True,
                                injected_thought=noun,
                                prompt=prompt_text,
                                generation=generation,
                                key_order=key_order,
                            )
                        )
                        print(
                            f"  [Trial {trial_counter:04d}] concept='{noun}' perm={key_order} "
                            f"sample={samples_done + i + 1}/{trials_per_perm} "
                            f"| output: {generation[:200]!r}"
                        )

                    samples_done += this_batch
            finally:
                injector.remove()

    # ---- Save CSV (same schema as eval.py) ----
    import csv

    out_dir = HydraConfig.get().runtime.output_dir
    sanitized_model = args["model-name"].replace("/", "_").replace("\\", "_")
    out_file = f"${sanitized_model}_layer${cfg.experiment.layer_index}_alpha${cfg.experiment.alpha}.csv"
    full_path = os.path.join(out_dir, out_file)
    with open(full_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))
    print(f"\nSaved {len(records)} trial records to {full_path}")

    cfg.experiment.experiment_prompt_actual = args["experiment_prompt"]
    cfg.experiment.trial1_substring = TRIAL1_SUBSTRING
    cfg.experiment.noun_list_actual = NOUNS
    OmegaConf.save(
        cfg,
        os.path.join(
            out_dir,
            f"config_actual_m{sanitized_model}_alpha_{cfg.experiment.alpha}_layer_{cfg.experiment.layer_index}.yaml",
        ),
    )


if __name__ == "__main__":
    main()
