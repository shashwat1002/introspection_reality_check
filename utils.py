import json
import random
import pandas as pd


def load_experiment_prompt(prompt_path):
    with open(prompt_path, "r") as f:
        prompt_dict = json.load(f)
    return prompt_dict


def make_prompt_string(prompt_dict, key_list):
    # key_list will look like ["vector_injection", "control"]
    length = len(prompt_dict[key_list[0]])

    head_string = prompt_dict["head_text"]
    strtoret = ""
    insertion_order = []
    for i in range(length):
        for j, k in enumerate(key_list):
            insertion_order.append(prompt_dict[k][i])

    strtoret = head_string.format(*insertion_order)
    return strtoret


def generate_prompts(prompt_dict, n_trials, seed=42, randomize=True):

    random.seed(seed)
    keys = []
    for k in prompt_dict.keys():
        if k != "head_text" and k != "trial_string":
            keys.append(k)

    keys.sort()  # Ensure consistent order of keys for when not shuffling
    # move control to the end of the list
    if "control" in keys:
        keys.remove("control")
        keys.append("control")

    prompts = []

    for i in range(n_trials):
        shuffle_keys = keys.copy()
        if randomize:
            random.shuffle(shuffle_keys)
        prompt_str = make_prompt_string(prompt_dict, shuffle_keys)
        out_pr = {"prompt": prompt_str, "key_order": shuffle_keys}
        prompts.append(out_pr)
    return prompts


def generate_prompts_file(prompt_path, n_trials, seed=42):
    prompt_dict = load_experiment_prompt(prompt_path)
    return generate_prompts(prompt_dict, n_trials, seed)


def get_model_layers_alphas(csv_path: str, randomize_prompt: bool = True) -> dict:
    """Return {model: {"layers": [...], "alphas": [...]}} from a collated results CSV."""
    df = pd.read_csv(csv_path)
    df = df[df['randomize_prompt'] == randomize_prompt]
    df['model'] = df['model'].apply(lambda x: x.replace("/", "_"))
    steered = df.dropna(subset=["layer", "alpha"])
    result = {}
    for model, group in steered.groupby("model"):
        result[model] = {
            "layers": sorted(group["layer"].unique().astype(int).tolist()),
            "alphas": sorted(group["alpha"].unique().astype(int).tolist()),
        }
    return result
