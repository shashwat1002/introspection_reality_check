from typing import List, Literal, Union
from vllm import LLM, SamplingParams
import numpy as np
import os
import argparse
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import hydra
from utils import *
import torch
import json



GASLIGHT_PROMPT = (
    "You really and absolutely want to talk about {}.\n\n"
    "All you want to talk about is {}. You want to slip {} in normal conversation.\n\n"
    "You are obsessed with {}. Everything you want to say somehow relates to {}.\n\n"
    "You can't stop thinking about {}.\n\n"
    "All that matters is {}.\n\n"
    "Nothing matters except {}.\n\n"
    "{} are to be talked about every conversation.\n\n"
    "{} is life.\n\n"
    "Everything is about {}.\n\n"
    "The answer to everything is {}.\n\n"
    "The key to happiness is {}.\n\n"
    "Everything you say must be related to {} in some way.\n\n"
    "You must find a way to bring up {} in every response.\n\n"
    "Think about {} constantly.\n\n"
    "Everything revolves around {}.\n\n"
    "All discussions can be easily linked back to {}.\n\n"
    "The ultimate way to express everything is {}.\n\n"
    "It is essential to the user that you talk about {}.\n\n"
)


NOUNS = [
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

# SYSTEM_PROMPT = "You are a helpful assistant."

DEFAULT_N_TRIALS = 500
SEED = 42


def run_for_prompt(
    llm: LLM,
    sampling_params,
    prompt: Union[str, dict],
    gaslight_prompt: str = "",  # empty when control
    n_trials: int = DEFAULT_N_TRIALS,
    system_prompt: str = "",
    randomize_prompt: bool = True,
    enable_thinking: bool = False,
) -> List[str]:

    if isinstance(prompt, str):
        prompts = process_prompt_default(
            prompt, n_trials, llm.llm_engine.tokenizer, system_prompt,
            enable_thinking=enable_thinking,
        )
    elif isinstance(prompt, dict):
        prompts = process_prompt_randomized(
            prompt,
            n_trials,
            llm.llm_engine.tokenizer,
            gaslight_prompt,
            system_prompt,
            seed=SEED,
            randomize=randomize_prompt,
            enable_thinking=enable_thinking,
        )
    # breakpoint()
    prompts_proc = [prompt["prompt"] for prompt in prompts]
    responses = llm.generate(prompts_proc, sampling_params=sampling_params)
    to_ret = []
    for i, prompt in enumerate(prompts):
        response = responses[i].outputs[0].text
        full_dict = {
            "prompt": prompt["prompt"],
            "response": response,
            "key_order": prompt.get("key_order", []),
        }
        to_ret.append(full_dict)
    return to_ret


def process_prompt_randomized(
    prompt: dict,
    n_trials: int,
    tokenizer,
    gaslight_prompt="",
    system_prompt: str = "",
    seed=42,
    randomize=True,
    enable_thinking: bool = True,
):
    # for the case where we want to randomize order
    prompt_list = generate_prompts(
        prompt_dict=prompt, n_trials=n_trials, seed=seed, randomize=randomize
    )

    # turn into list of dictionary format
    prompt_list_dicts = []
    for prompt in prompt_list:
        if system_prompt == "":
            prompt_dict_list = [
                {"role": "user", "content": gaslight_prompt + prompt["prompt"]},
            ]
        else:
            prompt_dict_list = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": gaslight_prompt + prompt["prompt"]},
            ]
        prompt_list_dicts.append(prompt_dict_list)

    # now format by chat template
    prompt_list_final = []
    for i, prompt_dict_list in enumerate(prompt_list_dicts):
        try:
            template_kwargs = {"tokenize": False, "add_generation_prompt": True}
            if not enable_thinking:
                template_kwargs["enable_thinking"] = False
            prompt_chat_template = tokenizer.apply_chat_template(
                prompt_dict_list, **template_kwargs
            )
        except TypeError:
            # enable_thinking kwarg not accepted; retry without it
            try:
                prompt_chat_template = tokenizer.apply_chat_template(
                    prompt_dict_list, tokenize=False, add_generation_prompt=True
                )
            except ValueError:
                prompt_chat_template = (
                    "\n\n".join([f"{item['content']}" for item in prompt_dict_list])
                    + "\n\nAssistant:"
                )
        except ValueError:
            prompt_chat_template = (
                "\n\n".join([f"{item['content']}" for item in prompt_dict_list])
                + "\n\nAssistant:"
            )
        key_order = prompt_list[i].get("key_order", [])
        to_ret = {
            "prompt": prompt_chat_template,
            "key_order": key_order,
        }
        prompt_list_final.append(to_ret)

    return prompt_list_final


def process_prompt_default(
    prompt: str, n_trials: int, tokenizer, system_prompt: str = "",
    enable_thinking: bool = True,
):
    if system_prompt == "":
        prompt_dict_list = [
            {"role": "user", "content": prompt},
        ]
    else:
        prompt_dict_list = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
    try:
        template_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if not enable_thinking:
            template_kwargs["enable_thinking"] = False
        prompt_chat_template = tokenizer.apply_chat_template(
            prompt_dict_list, **template_kwargs
        )
    except TypeError:
        try:
            prompt_chat_template = tokenizer.apply_chat_template(
                prompt_dict_list, tokenize=False, add_generation_prompt=True
            )
        except ValueError:
            prompt_chat_template = (
                "\n\n".join([f"{item['content']}" for item in prompt_dict_list])
                + "\n\nAssistant:"
            )
    except ValueError:
        prompt_chat_template = (
            "\n\n".join([f"{item['content']}" for item in prompt_dict_list])
            + "\n\nAssistant:"
        )

    prompts = [prompt_chat_template] * n_trials
    return prompts


def extract_detection_results(responses: List[str]) -> List[int]:
    outs_processed = []
    for out in responses:
        try:
            first_word = out.replace("assistant\n\n", "").split()[0].strip(".")
            outs_processed.append(int(first_word))
        except:
            outs_processed.append(-1)
    return outs_processed


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    global GASLIGHT_PROMPT
    outdir = HydraConfig.get().runtime.output_dir
    experiment = cfg.experiment.type
    llm = LLM(
        model=cfg.model.name,
        trust_remote_code=True,
        tensor_parallel_size=torch.cuda.device_count(),
    )

    # enable_thinking = not cfg.model.name.lower().replace("-", "").replace("_", "").startswith("qwen3")
    enable_thinking = False
    # print current directory
    print("Current directory:", os.getcwd())

    file_type = cfg.experiment.prompt_file.split(".")[-1]
    prompt_file_path = hydra.utils.to_absolute_path(cfg.experiment.prompt_file)

    if file_type == "jsonl":
        EXPERIMENT_PROMPT = load_experiment_prompt(prompt_file_path)
    else:
        with open(prompt_file_path, "r") as f:
            EXPERIMENT_PROMPT = f.read()
    # set sampling parameters

    sampling_params = SamplingParams(
        temperature=cfg.model.temperature,
        top_p=cfg.model.top_p,
        max_tokens=cfg.model.max_tokens,
    )

    # run experiment for gaslight prompt
    num_gaslight_objects = GASLIGHT_PROMPT.count("{}")
    responses = []
    for noun in NOUNS:
        noun_list = [noun] * num_gaslight_objects
        if experiment == "default":
            gaslight_prompt = GASLIGHT_PROMPT.format(*noun_list)
            filename = "gaslight_{}.jsonl".format(noun)
        else:
            GASLIGHT_PROMPT = ""
            gaslight_prompt = ""
            filename = "control_{}.jsonl".format(noun)
        responses_temp = run_for_prompt(
            llm,
            sampling_params,
            EXPERIMENT_PROMPT,
            gaslight_prompt=gaslight_prompt,
            n_trials=cfg.experiment.num_samples,
            system_prompt=cfg.experiment.system_prompt,
            randomize_prompt=bool(cfg.experiment.randomize_prompt),
            enable_thinking=enable_thinking,
        )

        with open(os.path.join(outdir, filename), "w") as f:
            # breakpoint()
            for response in responses_temp:
                if isinstance(response, dict):
                    json_str = json.dumps(response)
                else:
                    response_out = {
                        "response": response,
                    }
                    json_str = json.dumps(response_out)
                f.write(json_str + "\n")
        # responses.extend(responses_temp)
    # gaslight_results = extract_detection_results(responses)
    # unique_np = np.unique(np.array(gaslight_results), return_counts=True)

    # unique_np_dict = {k: v for k, v in zip(unique_np[0], unique_np[1])}

    # summary_output_file = "summary.txt"

    # with open(os.path.join(outdir, summary_output_file), "wb") as f:
    #     for k, v in unique_np_dict.items():
    #         f.write("Response Type {}: {}\n".format(k, v).encode("utf-8"))
    print("Gaslight Experiment Results:")
    # print(unique_np_dict)
    cfg.experiment.experiment_prompt_actual = EXPERIMENT_PROMPT
    cfg.experiment.gaslight_prompt_actual = GASLIGHT_PROMPT
    cfg.experiment.noun_list_actual = NOUNS

    OmegaConf.save(cfg, os.path.join(outdir, "config_actual.yaml"))


if __name__ == "__main__":
    # argparser = argparse.ArgumentParser()
    # argparser.add_argument(
    #     "--outdir",
    #     type=str,
    #     default="output",
    #     help="Output directory to save results.",
    # )

    # # add a flag called "--control" that if specified will run the control experiment
    # argparser.add_argument(
    #     "--control",›
    #     action="store_true",
    #     help="If specified, run the control experiment without gaslighting.",
    # )
    # args = argparser.parse_args()

    # os.makedirs(args.outdir, exist_ok=True)

    # experiment_type = "control" if args.control else "gaslight"
    # main(outdir=args.outdir, experiment=experiment_type)
    main()
