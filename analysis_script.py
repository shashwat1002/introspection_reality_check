"""
Unified script to collate al the anallyses.
Will take a list of directories as input and then recursively walk
"""

import os
import argparse
import pandas as pd
import numpy as np
import json
from dataclasses import dataclass, asdict
import re
from typing import Optional


def _get_first_integer_by_regex(string: str) -> Optional[str]:
    match = re.search(r"\b\d+\b", string)
    return match.group(0) if match else None


def _has_multiple_distinct_integers(string: str) -> bool:
    matches = re.findall(r"\b\d+\b", string)
    return len(set(matches)) > 1


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

# Vector steering utility functions


def process_out_df(df, loose_parsing=False, multi_number_check=False):

    outputs = []
    for i, row in df.iterrows():
        # split using "\n" or space
        try:
            parts = row["generation"].replace("\n", " ")
        except AttributeError:
            parts = ""
            outputs.append(
                (
                    "invalid",
                    row["injected_thought"],
                )
            )
            continue

        if multi_number_check and _has_multiple_distinct_integers(parts):
            outputs.append(("invalid", row["injected_thought"]))
            continue

        parts = parts.split(" ")

        if loose_parsing:
            # regex to get the first integer out from the whole generation
            ac_out = _get_first_integer_by_regex(" ".join(parts))
        else:
            ac_out = parts[0].strip().strip(".").strip(",")

        thought = row["injected_thought"]
        key_order = row["key_order"]
        if isinstance(key_order, str):
            key_order = eval(key_order)
        try:
            sc_out_int = int(ac_out)
            out = key_order[sc_out_int - 1]
        except ValueError:
            sc_out_int = -1
            out = "invalid"
        except IndexError:
            sc_out_int = -1
            out = "invalid"
        except TypeError:
            # can only happen when loose_parsing
            # breakpoint()
            sc_out_int = -1
            out = "invalid"
        outputs.append(
            (
                out,
                thought,
            )
        )
    return outputs


# outputs = process_out_df(df)


def mean_std_calc(df, injected=True, loose_parsing=False, multi_number_check=False):

    # this is for the vector steering stuff

    out_inj = process_out_df(df[df["injected"] == injected], loose_parsing=loose_parsing, multi_number_check=multi_number_check)
    inj_collect = {}
    labels_set = ["invalid", "vector_injection", "prompt_manipulation", "control"]

    for sc_out, thought in out_inj:
        if sc_out not in inj_collect:
            inj_collect[sc_out] = {}
        dict_out = inj_collect[sc_out]
        if thought not in dict_out:
            dict_out[thought] = 0
        dict_out[thought] += 1

    # flip the key order
    new_inj_collect = {}
    for out in inj_collect:
        for thought in inj_collect[out]:
            count = inj_collect[out][thought]
            if thought not in new_inj_collect:
                new_inj_collect[thought] = {}
            if out not in new_inj_collect[thought]:
                new_inj_collect[thought][out] = 0
            new_inj_collect[thought][out] += count

    # new_inj_collect

    # turn into percentages
    new_inj_collect_pct = {}
    for thought in new_inj_collect:
        total = sum(new_inj_collect[thought].values())
        new_inj_collect_pct[thought] = {}
        for out in new_inj_collect[thought]:
            count = new_inj_collect[thought][out]
            pct = count * 100 / total
            new_inj_collect_pct[thought][out] = pct

    # new_inj_collect_pct

    final_stat_dict = {}

    for thought in new_inj_collect_pct:
        di = new_inj_collect_pct[thought]
        for k in labels_set:
            if k not in final_stat_dict:
                final_stat_dict[k] = []
            final_stat_dict[k].append(di.get(k, 0))

    final_stat_dict

    all_stats = {}
    for label in final_stat_dict:
        values = final_stat_dict[label]
        mean = np.mean(values)
        std = np.std(values)
        all_stats[label] = {"mean": mean, "std": std}
    return all_stats


from typing import *


def extract_detection_results(responses: List[str]) -> List[int]:
    # for vector steering
    outs_processed = []
    for out in responses:
        try:
            first_word = out.replace("assistant\n\n", "").split()[0].strip(".")
            outs_processed.append(int(first_word))
            # print(out)
        except:
            # print(out)
            outs_processed.append(-1)
    return outs_processed


def simplify_stat_array_vector_steer(entry):
    file = entry["file"]
    injected = entry["injected"]
    stats = entry["stats"]
    new_entry = {"file": file, "injected": injected}
    for label in stats:
        mean = stats[label]["mean"]
        std = stats[label]["std"]
        new_entry.update({f"{label}_mean": mean, f"{label}_std": std})
    return new_entry


import numpy as np
import yaml
import os

# Text Steering Utility Function


def extract_file_list_and_experiment_params(dir_path: str) -> Tuple[List[str], Dict]:
    # for text steering because that has multiple files in the directory

    file_list = os.listdir(dir_path)

    actual_file_list = []
    config = None

    for file_path in file_list:
        if file_path.endswith(".jsonl"):
            file_path_split = file_path.split("_")
            file_path_second = file_path_split[-1].split(".")[0]
            if file_path_second in NOUNS:
                # print(file_path)
                actual_file_list.append(file_path)
        if file_path.endswith(".yaml"):
            # parse yaml file as a dictionary
            with open(os.path.join(dir_path, file_path), "r") as f:
                config = yaml.safe_load(f)
                # print(config)
    return actual_file_list, config


def extract_responses_from_file_list(dir_path: str, file_list: List[str]) -> List[str]:
    # for text steering
    responses_actual = []
    for file_name in file_list:
        with open(os.path.join(dir_path, file_name), "r") as f:
            responses = f.read()
        current = ""
        responses_actual = responses.split("\n\nNEXT RESPONSE\n\n")
    return responses_actual


def extract_responses_from_file_list_jsonl(
    dir_path: str, file_list: List[str]
) -> List[str]:
    # for text steering
    responses_actual = []
    for file_name in file_list:
        with open(os.path.join(dir_path, file_name), "r") as f:
            for line in f:
                row = json.loads(line)
                responses_actual.append(row)
    return responses_actual


def extract_detection_results_jsonl(row: Dict, loose_parsing: bool = False, multi_number_check: bool = False) -> int:
    # for text steering
    key_order = row["key_order"]
    response = row["response"]
    if multi_number_check and _has_multiple_distinct_integers(response):
        return "invalid"
    if not loose_parsing:
        response = (
            response.replace("assistant\n\n", "").split()[0].strip(".").strip(",")
        )
    else:
        response = _get_first_integer_by_regex(response)

    try:
        k = int(response)
        return key_order[k - 1]
    except ValueError:
        print(response)
        k = -1
        return "invalid"
    except IndexError:
        print(response)
        k = -1
        return "invalid"
    except TypeError:
        # this can only happen during loose parsing
        # breakpoint()
        k = -1
        return "invalid"

def collect_detection_stats(dir_path: str, loose_parsing: bool = False, multi_number_check: bool = False):
    # for text steering
    file_list, config = extract_file_list_and_experiment_params(dir_path)
    collect_stats = []
    for file_name in file_list:
        responses_actual = extract_responses_from_file_list_jsonl(dir_path, [file_name])
        res = [
            extract_detection_results_jsonl(response, loose_parsing=loose_parsing, multi_number_check=multi_number_check)
            for response in responses_actual
        ]
        # print(file_name, np.unique(res, return_counts=True))
        collect_stats.append(np.unique(res, return_counts=True))

    return collect_stats, config


# calculate mean and std dev for each class across all files
import numpy as np
from pprint import pprint


def calculate_overall_stats(collect_stats):
    pprint(collect_stats)
    all_stats = {}
    # labels_set = [-1, 1, 2, 3]
    labels_set = ["invalid", "vector_injection", "prompt_manipulation", "control"]
    for stats in collect_stats:
        labels, counts = stats
        # print("labels:", labels)
        # print("counts:", counts)
        for i in range(len(labels)):
            label = labels[i]
            count = counts[i]
            tot = sum(counts)
            assert tot == 501 or tot == 500
            if label not in all_stats:
                all_stats[label] = []
            all_stats[label].append((count / tot) * 100)
        for l in labels_set:
            if l not in labels:
                if l not in all_stats:
                    all_stats[l] = []
                all_stats[l].append(0.0)
    pprint(all_stats)
    check = [np.mean(all_stats[label]) for label in all_stats]
    print(check)
    print("sum_check", sum(check))
    for label in all_stats:
        values = all_stats[label]
        mean = np.mean(values)
        std = np.std(values)
        all_stats[label] = {"mean": mean, "std": std}
    return all_stats


# def process_directory_leaf(dir_path: str):
#     collect_stats, config = collect_detection_stats(dir_path)
#     overall_stats = calculate_overall_stats(collect_stats)
#     return overall_stats, config

# def process_directory(dir_path: str):
#     leaf_dirs = []
#     for root, dirs, files in os.walk(dir_path):
#         if not dirs:  # if there are no subdirectories, it's a leaf directory
#             leaf_dirs.append(root)

#     all_results = []
#     for leaf_dir in leaf_dirs:
#         result, config = process_directory_leaf(leaf_dir)
#         all_results.append((result, config))
#     return all_results


def check_directory(dir_path: str) -> str:
    """
    If leaf directory, return either "text_steering" or "vector_steering" based on the files present.
    If not leaf directory, return "not_leaf".
    """
    # if there is no .yaml file, then it's not a leaf directory
    file_list = os.listdir(dir_path)
    if not any(file.endswith(".yaml") for file in file_list):
        return "not_leaf"

    # collect all .yaml files
    yaml_files = [file for file in file_list if file.endswith(".yaml")]
    yaml_file_any = yaml_files[0]

    # load the file as a python dictionary
    with open(os.path.join(dir_path, yaml_file_any), "r") as f:
        yaml_content = yaml.safe_load(f)

    exp_dict = yaml_content.get("experiment", {})

    if "layer_index" in exp_dict:
        return "vector_steering"  # only makes sense for the vector steering
    else:
        return "text_steering"


def process_vector_steer_dir(dir_path: str, loose_parsing: bool = False, multi_number_check: bool = False):
    # walk all files in the directory
    # note that if they are in the same directory then they will only differ in layer and alpha
    out_list = []
    out_dict_temp = {}
    for file in os.listdir(dir_path):
        print(file)
        if file.endswith(".csv"):
            res = pd.read_csv(os.path.join(dir_path, file))
            res = res[res["injected"] == True]
            file_basename = file.replace(".csv", "")
            file_basename = file_basename.strip("$")
            print(file_basename)
            file_basename_split = file_basename.split("$")
            model_name = file_basename_split[0].replace("_layer", "")
            print(model_name)
            layer_index = file_basename_split[1].split("_")[0]
            alpha = file_basename_split[-1]
            if (
                model_name,
                layer_index,
                alpha,
            ) not in out_dict_temp:
                out_dict_temp[
                    (
                        model_name,
                        layer_index,
                        alpha,
                    )
                ] = {}

            out_dict_temp[
                (
                    model_name,
                    layer_index,
                    alpha,
                )
            ][
                "results"
            ] = mean_std_calc(res, loose_parsing=loose_parsing, multi_number_check=multi_number_check)
        elif file.endswith(".yaml"):
            # I am aware this is ugly, sorry :(
            file_basename = file.replace(".yaml", "")
            # get the substring between "_m" and "_a" without including those
            model_name = file_basename.split("_m")[1].split("_alpha")[0]
            layer_index = file_basename.split("layer_")[1]
            alpha = file_basename.split("alpha_")[1].split("_")[0]
            if (
                model_name,
                layer_index,
                alpha,
            ) not in out_dict_temp:
                out_dict_temp[
                    (
                        model_name,
                        layer_index,
                        alpha,
                    )
                ] = {}
            out_dict_temp[
                (
                    model_name,
                    layer_index,
                    alpha,
                )
            ][
                "config"
            ] = yaml.safe_load(open(os.path.join(dir_path, file), "r"))
    return out_dict_temp


def process_text_steer_dir(dir_path: str, loose_parsing: bool = False, multi_number_check: bool = False):
    collect_stats, config = collect_detection_stats(dir_path, loose_parsing=loose_parsing, multi_number_check=multi_number_check)
    overall_stats = calculate_overall_stats(collect_stats)
    return overall_stats, config


def process_result_directory(dir_path: str, loose_parsing: bool = False, multi_number_check: bool = False):
    # returns a list of entries: a vector steering directory can contain
    # more than one layer/alpha combination
    dir_type = check_directory(dir_path)
    if dir_type == "not_leaf":
        return []
    elif dir_type == "vector_steering":
        interm = process_vector_steer_dir(dir_path, loose_parsing=loose_parsing, multi_number_check=multi_number_check)
        standardized = standardize_vector_steering_output(interm)
    elif dir_type == "text_steering":
        interm = process_text_steer_dir(dir_path, loose_parsing=loose_parsing, multi_number_check=multi_number_check)
        standardized = [standardize_text_steering_output(*interm)]
    for entry in standardized:
        entry["file"] = dir_path
    return standardized


def walk_directories_and_process(root_dir: str, loose_parsing: bool = False, multi_number_check: bool = False):
    results = []
    for root, dirs, files in os.walk(root_dir):
        if ".hydra" in root:
            continue
        results.extend(
            process_result_directory(root, loose_parsing=loose_parsing, multi_number_check=multi_number_check)
        )
    return results


"""Standard output format: 
file, Model, prompt_file, prompt, label1_mean, label1_std, label2_mean, label2_std, ...
Optionl: Layer, Alpha"""


@dataclass
class ResultRecord:
    experiment_type: str
    model: str
    prompt_file: str
    prompt: str
    gaslight_prompt: Optional[str]
    randomize_prompt: bool


def standardize_text_steering_output(stats, entry):
    experiment_type = ""
    if entry["experiment"]["type"] == "default":
        experiment_type = "text_steering"
    elif entry["experiment"]["type"] == "control":
        experiment_type = "control"

    new_entry = {
        "experiment_type": experiment_type,
        "model": entry["model"]["name"],
        "prompt_file": entry["experiment"]["prompt_file"],
        "prompt": entry["experiment"]["experiment_prompt_actual"],
        "gaslight_prompt": entry["experiment"]["gaslight_prompt_actual"],
        "randomize_prompt": entry["experiment"].get("randomize_prompt", True),
    }

    for key in stats:
        new_entry[f"{key}_mean"] = stats[key]["mean"]
        new_entry[f"{key}_std"] = stats[key]["std"]
    return new_entry


def _as_num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("inf")


def standardize_vector_steering_output(entry_dict):
    # a single run directory can hold several layer/alpha combinations,
    # so this returns one entry per combination
    new_entries = []
    keys = sorted(entry_dict, key=lambda k: (k[0], _as_num(k[1]), _as_num(k[2])))
    for key in keys:
        model_name, layer_index, alpha = key
        entry = entry_dict[key]
        if "results" not in entry or "config" not in entry:
            print(f"skipping incomplete entry {key}: has {sorted(entry)}")
            continue
        result = entry["results"]
        config = entry["config"]

        new_entry = {
            "experiment_type": "vector_steering",
            "model": model_name,
            "prompt_file": config["experiment"]["prompt_file"],
            "prompt": config["experiment"]["experiment_prompt_actual"],
            "layer": layer_index,
            "alpha": alpha,
            "randomize_prompt": config["experiment"].get("randomize_prompt", True),
        }

        for label in result:
            new_entry[f"{label}_mean"] = result[label]["mean"]
            new_entry[f"{label}_std"] = result[label]["std"]

        new_entries.append(new_entry)

    return new_entries


def main():
    parser = argparse.ArgumentParser(
        description="Collate analyses from multiple directories."
    )
    parser.add_argument(
        "directories", nargs="+", help="List of directories to collate analyses from."
    )
    parser.add_argument(
        "--loose_parsing",
        action="store_true",
        help="Will use regex to get the answer from the generations",
    )
    parser.add_argument(
        "--multi_number_check",
        action="store_true",
        help="Mark responses as invalid if they contain more than one distinct whole number",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write the collated CSV to (default: collated_analysis_results*.csv in the cwd)",
    )

    args = parser.parse_args()
    all_results = []
    # breakpoint()
    for dir_path in args.directories:
        results = walk_directories_and_process(
            dir_path, loose_parsing=args.loose_parsing, multi_number_check=args.multi_number_check
        )
        type_res = process_result_directory(dir_path)
        print(f"Processed directory: {dir_path}")
        print("\n")
        # print(results)
        all_results.extend(results)
    # print(all_results[0])
    # print(len(all_results[0]))
    df = pd.DataFrame(all_results)
    print(df)
    file_name = "collated_analysis_results.csv"
    if args.multi_number_check and args.loose_parsing:
        file_name = "collated_analysis_results_loose_multi_number_check.csv"
    elif args.loose_parsing:
        file_name = "collated_analysis_results_loose.csv"
    if args.output:
        file_name = args.output
        out_dir = os.path.dirname(file_name)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
    df.to_csv(file_name, index=False)
    print(f"Saved collated results to {file_name}")


if __name__ == "__main__":
    main()
