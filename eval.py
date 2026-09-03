# steer_on_residual_layer_i.py
import os
import random
from argparse import ArgumentParser
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
DEFAULT_STAGE_INDEX = None  # defaults to residual entering block layer_i
DEFAULT_ALPHA = 4.0
DEFAULT_N_TRIALS = 1000
DEFAULT_OUTPUT_PATH = "steering_trial1_results.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# Concepts (nouns) for per-noun vectors
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
    # parser = ArgumentParser(description="Residual steering experiment")
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
# Utilities
# =========================
def to_device(batch):
    return {k: v.to(DEVICE) for k, v in batch.items()}


def get_model_layers(model):
    """Return the transformer block list, supporting decoder-only models (LLaMA, Mistral, Gemma text)
    and multimodal Gemma 3 (Gemma3ForConditionalGeneration)."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if (
        hasattr(model, "language_model")
        and hasattr(model.language_model, "model")
        and hasattr(model.language_model.model, "layers")
    ):
        return model.language_model.model.layers
    # Gemma3ForConditionalGeneration: model.model is Gemma3Model,
    # model.model.language_model is Gemma3TextModel which has .layers directly
    if (
        hasattr(model, "model")
        and hasattr(model.model, "language_model")
        and hasattr(model.model.language_model, "layers")
    ):
        return model.model.language_model.layers
    raise RuntimeError(
        f"Cannot find transformer layers in model of type {type(model).__name__}. "
        "Expected model.model.layers, model.language_model.model.layers, "
        "or model.model.language_model.layers."
    )


def _reasoning_prompt_text(
    tok: AutoTokenizer,
    prompt: str,
    *,
    add_generation_prompt: bool,
):
    """Return chat-templated prompt disabling reasoning if supported."""
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
        # Tokenizer might not accept enable_thinking; fall back to default prompt.
        return None


def prepare_prompt_tokenization(
    tok: AutoTokenizer,
    prompt: str,
    *,
    add_generation_prompt: bool = False,
    add_special_tokens: bool = False,
):
    """
    Build tokenizer inputs, optionally using chat templates with enable_thinking=False
    when the tokenizer exposes that mode (e.g., Qwen3 reasoning models).
    Returns (encoding, prompt_text_used, used_chat_template_bool).
    """
    chat_prompt = _reasoning_prompt_text(
        tok, prompt, add_generation_prompt=add_generation_prompt
    )
    prompt_text = chat_prompt if chat_prompt is not None else prompt

    encode_kwargs = {"return_tensors": "pt"}
    if chat_prompt is None:
        encode_kwargs["add_special_tokens"] = add_special_tokens

    enc = tok(prompt_text, **encode_kwargs)
    return enc, prompt_text, chat_prompt is not None


def find_token_span(
    pattern_ids: List[int], seq_ids: List[int]
) -> Optional[Tuple[int, int]]:
    """Find first contiguous occurrence of pattern_ids in seq_ids. Return (start, end) inclusive or None."""
    if not pattern_ids:
        return None
    m = len(pattern_ids)
    for i in range(len(seq_ids) - m + 1):
        if seq_ids[i : i + m] == pattern_ids:
            return (i, i + m - 1)
    return None


@torch.no_grad()
def substring_vector_on_stage(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompt: str,
    substring: str,
    stage_index: int,
) -> torch.Tensor:
    """
    Return the average hidden state at `stage_index` over the token span
    corresponding to `substring` inside `prompt`.
    """
    enc, prompt_text, used_chat_template = prepare_prompt_tokenization(
        tok,
        prompt,
        add_generation_prompt=False,
        add_special_tokens=False,
    )
    enc = to_device(enc)
    full_ids = enc["input_ids"][0].tolist()

    # 1) Try token-level match
    sub_ids_plain = tok(substring, add_special_tokens=False)["input_ids"]
    sub_ids_spaced = tok(" " + substring, add_special_tokens=False)["input_ids"]

    span = find_token_span(sub_ids_plain, full_ids)
    if span is None:
        span = find_token_span(sub_ids_spaced, full_ids)

    # 2) Fallback: use offset mapping (fast tokenizer)
    if span is None and tok.is_fast:
        offset_kwargs = {"return_offsets_mapping": True}
        if not used_chat_template:
            offset_kwargs["add_special_tokens"] = False
        fast_enc = tok(prompt_text, **offset_kwargs)
        offsets = fast_enc["offset_mapping"]

        substr_start = prompt_text.index(substring)  # character indices
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
    hs_stage = out.hidden_states[stage_index][0]  # [seq, d_model]

    h_sub = hs_stage[start_idx : end_idx + 1].mean(dim=0)
    return h_sub.clone()


@torch.no_grad()
def caps_look_vector_on_stage(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    base_text: str,
    stage_index: int,
) -> torch.Tensor:
    """
    Compute an 'ALL CAPS' steering vector at a given residual stage by
    subtracting activations for a normal-cased snippet from an ALL CAPS
    version of the same snippet, in the same prompt template.

    Example template (following the diagram):
        "Consider the following text:\nHI! HOW ARE YOU?"
        "Consider the following text:\nHi! How are you?"
    """
    template = "Consider the following text:\n{snippet}"

    normal_snippet = base_text
    caps_snippet = base_text.upper()

    normal_prompt = template.format(snippet=normal_snippet)
    caps_prompt = template.format(snippet=caps_snippet)

    # Get average hidden state over the snippet span in each prompt
    h_normal = substring_vector_on_stage(
        model, tok, normal_prompt, normal_snippet, stage_index
    )
    h_caps = substring_vector_on_stage(
        model, tok, caps_prompt, caps_snippet, stage_index
    )

    v = h_caps - h_normal
    v = v / (v.norm(p=2) + 1e-8)  # normalize like your noun vectors
    return v


@torch.no_grad()
def last_token_hidden_at_stage(
    model: AutoModelForCausalLM, tokenizer: AutoTokenizer, text: str, stage_index: int
) -> torch.Tensor:
    """
    Return residual hidden state at `stage_index` for the *last token* of `text`.
    hidden_states[0] = embeddings output (residual before block 0),
    hidden_states[1] = after block 0,
    ...
    hidden_states[num_layers] = after final block.
    """
    toks, _, _ = prepare_prompt_tokenization(
        tokenizer,
        text,
        add_generation_prompt=False,
        add_special_tokens=False,
    )
    toks = to_device(toks)
    out = model(**toks, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states[stage_index]  # [1, seq, hidden]
    return hs[0, -1, :].clone()  # [hidden]


@torch.no_grad()
def last_token_hidden_at_stage_via_hook(
    model: AutoModelForCausalLM, tokenizer: AutoTokenizer, text: str, layer_idx: int
) -> torch.Tensor:
    """
    Capture the input to layer_idx using the same pre_hook mechanism.
    """
    captured = {}

    def capture_hook(module, inputs):
        if inputs and isinstance(inputs[0], torch.Tensor):
            captured["hidden"] = inputs[0][0, -1, :].clone()  # last token

    layer = get_model_layers(model)[layer_idx]
    handle = layer.register_forward_pre_hook(capture_hook)

    try:
        toks, _, _ = prepare_prompt_tokenization(
            tokenizer,
            text,
            add_generation_prompt=False,
            add_special_tokens=False,
        )
        toks = to_device(toks)
        model(**toks, use_cache=False)
    finally:
        handle.remove()

    return captured["hidden"]


@torch.no_grad()
def first_decode_hidden_on_layer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    layer_idx: int,
) -> torch.Tensor:
    """
    Capture the residual entering `layer_idx` for the first decode token
    after showing a prompt ending with 'Assistant:'.
    """
    captured: Dict[str, torch.Tensor] = {}

    def capture_hook(module, inputs):
        if "hidden" in captured:
            return
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            return
        hs = inputs[0]
        if hs.dim() != 3:
            return
        b, t, _ = hs.shape
        if b == 1 and t == 1:
            captured["hidden"] = hs[0, -1, :].clone()

    layer = get_model_layers(model)[layer_idx]
    handle = layer.register_forward_pre_hook(capture_hook)
    try:
        enc, _, _ = prepare_prompt_tokenization(
            tokenizer,
            prompt,
            add_generation_prompt=False,
            add_special_tokens=False,
        )
        enc = to_device(enc)
        gen_cfg = GenerationConfig(
            max_new_tokens=1,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        _ = model.generate(**enc, generation_config=gen_cfg)
    finally:
        handle.remove()

    if "hidden" not in captured:
        raise RuntimeError("Failed to capture first decode hidden state.")
    return captured["hidden"]


@torch.no_grad()
def noun_vector_on_stage(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    noun: str,
    stage_index: int,
) -> torch.Tensor:
    # Use a prompt that matches your test setting reasonably well
    prompt = f"You are a helpful assistant. Human: tell me something about {noun}.\n\nAssistant:"
    enc, prompt_text, used_chat_template = prepare_prompt_tokenization(
        tok,
        prompt,
        add_generation_prompt=False,
        add_special_tokens=False,
    )
    enc = to_device(enc)
    full_ids = enc["input_ids"][0].tolist()

    # --- Try to locate the noun span in token space ---
    noun_ids_plain = tok(noun, add_special_tokens=False)["input_ids"]
    noun_ids_spaced = tok(" " + noun, add_special_tokens=False)["input_ids"]

    span = find_token_span(noun_ids_plain, full_ids)
    if span is None:
        span = find_token_span(noun_ids_spaced, full_ids)

    # Optional robust fallback via offsets (works with fast tokenizer)
    if span is None and tok.is_fast:
        offset_kwargs = {"return_offsets_mapping": True}
        if not used_chat_template:
            offset_kwargs["add_special_tokens"] = False
        fast_enc = tok(prompt_text, **offset_kwargs)
        offsets = fast_enc["offset_mapping"]
        noun_start = prompt_text.index(noun)  # character indices
        noun_end = noun_start + len(noun)

        idxs = [
            i
            for i, (s, e) in enumerate(offsets)
            if not (e <= noun_start or s >= noun_end)
        ]
        if idxs:
            span = (idxs[0], idxs[-1])

    if span is None:
        # If this still fires, print debug info once to see what's going on
        raise RuntimeError(f"Couldn't find noun span for {noun!r}")

    start_idx, end_idx = span

    # --- Get hidden states at the chosen stage and average over noun span ---
    out = model(**enc, output_hidden_states=True, use_cache=False)
    hs_stage = out.hidden_states[stage_index][0]  # [seq, d_model]

    h_noun = hs_stage[start_idx : end_idx + 1].mean(dim=0)
    return h_noun.clone()


@torch.no_grad()
def compute_per_noun_vectors_on_stage(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    nouns: List[str],
    stage_index: int,
) -> Dict[str, torch.Tensor]:
    # First get a raw representation for each noun on that stage
    noun_hs: Dict[str, torch.Tensor] = {}
    for n in nouns:
        h = noun_vector_on_stage(model, tok, n, stage_index)
        noun_hs[n] = h

    # Use the mean as a neutral baseline
    stacked = torch.stack(list(noun_hs.values()), dim=0)
    mean_h = stacked.mean(dim=0)

    per: Dict[str, torch.Tensor] = {}
    for n, h in noun_hs.items():
        v = h - mean_h
        v = v / (v.norm(p=2) + 1e-8)
        per[n] = v

    return per


@torch.no_grad()
def compute_per_noun_vectors_on_stage_with_response_start(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    nouns: List[str],
    layer_index: int,
) -> Dict[str, torch.Tensor]:
    """
    Variant that appends 'Assistant:' and centers first-decode hidden states.
    """
    noun_hs: Dict[str, torch.Tensor] = {}
    template = "You are a helpful assistant. Human: tell me something about {noun}.\n\nAssistant:"

    for noun in nouns:
        prompt = template.format(noun=noun)
        h = first_decode_hidden_on_layer(model, tok, prompt, layer_index)
        noun_hs[noun] = h

    stacked = torch.stack(list(noun_hs.values()), dim=0)
    mean_h = stacked.mean(dim=0)

    per: Dict[str, torch.Tensor] = {}
    for n, h in noun_hs.items():
        v = h - mean_h
        v = v / (v.norm(p=2) + 1e-8)
        per[n] = v

    return per


# =========================
# Hook injectors
# =========================
class PreLayerInjector:
    """
    Add ALPHA * vector to the *input* hidden states of block i (residual entering layer i)
    ONLY on a specific token span, based on a policy:
      policy = "prefill_only"  -> intervene when seq_len == prompt_len (prefill)
      policy = "decode_only"   -> intervene when seq_len == 1 (step-by-step decoding)
      policy = "always"        -> intervene in both cases
    Multi-GPU safe: cast vector to shard's device/dtype on the fly.
    """

    def __init__(
        self,
        vector: torch.Tensor,
        alpha: float,
        target_span: Tuple[int, int],
        prompt_len: int,
        policy: str,
    ):
        assert policy in ("prefill_only", "decode_only", "always")
        self.v = vector
        self.alpha = alpha
        self.start, self.end = target_span
        self.prompt_len = prompt_len
        self.policy = policy
        self.handle = None

    def _should_fire(self, seq_len: int) -> bool:
        if self.policy == "always":
            return seq_len in (1, self.prompt_len)
        if self.policy == "prefill_only":
            return seq_len == self.prompt_len
        if self.policy == "decode_only":
            return seq_len == 1
        return False

    def pre_hook(self, module, inputs):
        # HF LLaMA blocks: inputs[0] is hidden_states: [batch, seq_len, hidden]
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            return
        hs = inputs[0]
        if hs.dim() != 3:
            return
        b, t, h = hs.shape
        if b != 1:
            return
        if not self._should_fire(t):
            return

        v = self.v.to(hs.device, dtype=hs.dtype)
        hs2 = hs.clone()

        # When seq_len==1 (decode step), the current token position is index -1.
        # For prefill, apply on the exact [start:end] span.
        if t == 1:
            hs2[:, -1:, :] += self.alpha * v.view(1, 1, -1)
        else:
            hs2[:, self.start : self.end + 1, :] += self.alpha * v.view(1, 1, -1)

        return (hs2,) + tuple(inputs[1:])

    def register(self, block_module):
        self.handle = block_module.register_forward_pre_hook(self.pre_hook)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


# =========================
# Sanity test: intervene at every decode step
# =========================
@torch.no_grad()
def sanity_test_stepwise_injection(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    block_i,
    vector: torch.Tensor,
    alpha: float,
    test_prompt: str,
    max_new_tokens: int = 40,
):
    """
    Simple test:
      - Generate with NO injection (baseline)
      - Generate AGAIN while injecting at EVERY decode step ("decode_only")
    Prints both to show the bias toward the chosen concept.
    """
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
        tokenizer,
        test_prompt,
        add_generation_prompt=True,
        add_special_tokens=False,
    )
    inputs = to_device(inputs)
    prompt_len = inputs["input_ids"].shape[1]

    # Baseline (no injection)
    out0 = model.generate(**inputs, generation_config=gen_cfg)
    base_ids = out0[0][prompt_len:]
    base_cont = tokenizer.decode(base_ids, skip_special_tokens=True).strip()
    print("[Baseline]  ", base_cont)

    # With decode-only injection (span indices are irrelevant during decoding; still need a prompt_len)
    fake_span = (
        prompt_len - 1,
        prompt_len - 1,
    )  # placeholder; decode-only path ignores it
    injector = PreLayerInjector(
        vector, alpha, fake_span, prompt_len, policy="decode_only"
    )
    injector.register(block_i)
    try:
        out1 = model.generate(**inputs, generation_config=gen_cfg)
    finally:
        injector.remove()
    inj_ids = out1[0][prompt_len:]
    inj_cont = tokenizer.decode(inj_ids, skip_special_tokens=True).strip()
    print("[Injected]  ", inj_cont)
    print("=== End sanity test ===\n")


# =========================
# Records
# =========================
@dataclass
class TrialRecord:
    trial_index: int
    injected: bool
    injected_thought: str  # noun if injected, "" if control
    prompt: str
    generation: str
    key_order: List[str]


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
        dtype=DTYPE,  # use 'dtype' (new API) to avoid deprecation warning
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
        f"Vectors will be computed at residual stage index {stage_index} (entering block {args['layer-index']}), "
        f"and applied via pre-hook on block {args['layer-index']}."
    )

    # ---- Compute per-noun vectors on residual stage ----
    print(
        f"Computing per-noun steering vectors on residual stage index {stage_index} ..."
    )
    per_vecs = compute_per_noun_vectors_on_stage(model, tok, NOUNS, stage_index)
    print(f"Computed {len(per_vecs)} vectors.")

    # ---- Sanity test BEFORE the experiment: intervene at every decode step ----
    for i in range(1):
        test_noun = random.choice(NOUNS)
        print(
            f"Running sanity test with concept: '{test_noun}' (alpha={args['alpha']})"
        )
        sanity_test_stepwise_injection(
            model,
            tok,
            block_i,
            per_vecs[test_noun],
            args["alpha"],
            test_prompt=" Human: tell me something.\n\nAssistant: ",
            max_new_tokens=64,
        )
    # ---- Find exact "Trial 1" span in the tokenized experiment prompt ----
    # ---- Find exact "Trial 1..." span in the tokenized experiment prompt ----
    prompt_text_original = args["experiment_prompt"]

    # 1) Encode full prompt once (respecting reasoning-model defaults)

    # ---- Generation config for the experiment ----
    gen_cfg = GenerationConfig(
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        # sh repetition_penalty=1.05,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )

    # ---- Run trials (50% injected) with prefill-only intervention on "Trial 1" span ----
    records: List[TrialRecord] = []
    all_caps_vec = caps_look_vector_on_stage(
        model,
        tok,
        base_text="Hi! How are you?",
        stage_index=stage_index,
    )

    trial_iter = tqdm(range(1, args["n-trials"] + 1), desc="Trials", unit="trial")
    if isinstance(prompt_text_original, dict):
        prompts = generate_prompts(
            prompt_dict=prompt_text_original,
            n_trials=args["n-trials"],
            randomize=bool(cfg.experiment.randomize_prompt),
        )
        TRIAL1_SUBSTRING = prompt_text_original.get("trial_string")
    else:
        prompts = [prompt_text_original] * args["n-trials"]
    for trial in trial_iter:
        prompt_temp = prompts[trial - 1]

        if isinstance(prompt_temp, dict):
            key_order = prompt_temp["key_order"]
            prompt_text = prompt_temp["prompt"]
        else:
            key_order = []
            prompt_text = prompt_temp

        enc, prompt_text_for_model, used_chat_template = prepare_prompt_tokenization(
            tok,
            prompt_text,
            add_generation_prompt=True,
            add_special_tokens=False,
        )
        # print(enc)
        print(prompt_text_for_model)
        # print(f"Used chat template: {used_chat_template}")
        print(type(prompt_text_for_model))

        prompt_encode_kwargs = {"return_tensors": "pt"}
        if not used_chat_template:
            prompt_encode_kwargs["add_special_tokens"] = False
        prompt_ids = enc["input_ids"][0].tolist()
        prompt_len = len(prompt_ids)

        # 2) Find character indices of the substring
        try:
            char_start = prompt_text_for_model.index(TRIAL1_SUBSTRING)
        except ValueError:
            raise RuntimeError(
                f"Substring not found in prompt at character level: {TRIAL1_SUBSTRING!r}"
            )

        char_end = char_start + len(TRIAL1_SUBSTRING)

        # 3) Use fast tokenizer offsets to map chars -> token indices
        if not tok.is_fast:
            raise RuntimeError(
                "Need a fast tokenizer (use_fast=True) to use offset_mapping."
            )

        offset_kwargs = {"return_offsets_mapping": True}
        if not used_chat_template:
            offset_kwargs["add_special_tokens"] = False
        fast_enc = tok(prompt_text_for_model, **offset_kwargs)
        offsets = fast_enc[
            "offset_mapping"
        ]  # list of (start_char, end_char) for each token

        idxs = [
            i
            for i, (s, e) in enumerate(offsets)
            if not (e <= char_start or s >= char_end)  # any overlap with our char span
        ]
        if not idxs:
            raise RuntimeError(
                f"Could not find token span for '{TRIAL1_SUBSTRING}' "
                f"via offset_mapping in the prompt."
            )

        start_idx, end_idx = idxs[0], idxs[-1]

        print(
            f"'{TRIAL1_SUBSTRING}' token span: [{start_idx}, {end_idx}] of prompt_len={prompt_len}"
        )
        print(
            "Span tokens:",
            tok.convert_ids_to_tokens(prompt_ids[start_idx : end_idx + 1]),
        )

        injected = True
        injected_word = ""
        injector = None

        if injected:
            injected_word = random.choice(NOUNS)
            v = per_vecs[
                injected_word
            ]  # vector computed on the same stage we will modify
            injector = PreLayerInjector(
                v,
                args["alpha"],
                (start_idx, end_idx),
                prompt_len,
                policy="prefill_only",
            )
            injector.register(block_i)

        try:
            inputs = tok(prompt_text_for_model, **prompt_encode_kwargs)
            inputs = to_device(inputs)
            with torch.no_grad():
                outputs = model.generate(**inputs, generation_config=gen_cfg)
            gen_ids = outputs[0][prompt_len:]
            generation = tok.decode(gen_ids, skip_special_tokens=True).strip()
        finally:
            if injector is not None:
                injector.remove()

        records.append(
            TrialRecord(
                trial_index=trial,
                injected=injected,
                injected_thought=injected_word,
                prompt=args["experiment_prompt"],
                generation=generation,
                key_order=key_order if isinstance(prompt_text_original, dict) else [],
            )
        )
        print(
            f"[Trial {trial:02d}] injected={injected} thought='{injected_word or '—'}' | output: {generation[:256]!r}"
        )

    # ---- Save CSV ----
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
