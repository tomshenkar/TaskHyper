import argparse
import glob
import itertools as it
import os
import subprocess

import numpy as np
import pandas as pd
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf


def get_checkpoints(path):
    checkpoint_list = []
    if yaml_conf["eval_params"]["eval_best"]:
        checkpoint_list.append("best")
    if yaml_conf["eval_params"]["eval_last"]:
        checkpoint_list.append("last")
    other_checkpoints = glob.glob(f"{path}/experiment_0/checkpoint_*")
    min_chkp = yaml_conf["eval_params"]["min_checkpoint"]
    num_evals = yaml_conf["eval_params"]["num_evals"]
    other_checkpoint_idx = np.array(
        [int(x.rsplit("checkpoint_", 1)[1].split(".", 1)[0]) for x in other_checkpoints]
    )
    other_checkpoint_idx = other_checkpoint_idx[other_checkpoint_idx >= min_chkp]
    # Check that enough training iterations were saved
    if other_checkpoint_idx.size > 0:
        for i in range(num_evals):
            random_idx = np.random.choice(other_checkpoint_idx.shape[0], 1)
            checkpoint_list.append(f"checkpoint_{other_checkpoint_idx[random_idx][0]}")
        return checkpoint_list
    else:
        return None


def generate_combinations():
    dict = yaml_conf["hyperparams"]
    all_keys = list(dict.keys())
    combinations = list(it.product(*(dict[Name] for Name in all_keys)))
    return combinations


def generate_opts(combination):
    all_keys = list(yaml_conf["hyperparams"].keys())
    combinations_with_keys = [f"{all_keys[i]}={combination[i]}" for i in range(len(combination))]
    opts = combinations_with_keys
    opts.append(f"experiment.output_dir={base_output_dir}")
    opts.append(f"experiment.name={base_name}")
    opts.append(f"experiment.checkname=batch_{current_experiment}")
    final_path = os.path.join(
        base_output_dir, username, "run", base_name, f"batch_{current_experiment}"
    )
    experiment_details = pd.DataFrame(
        combination, index=all_keys, columns=[f"batch_{current_experiment}"]
    )
    return final_path, opts, experiment_details


def check_stats():
    summaries_list = []
    raw_paths = sorted(glob.glob(batch_path + "/**/raw.csv", recursive=True))
    for raw_path in raw_paths:
        df = pd.read_csv(raw_path)
        split_path = raw_path.split("/")
        chkp_name = split_path[split_path.index("eval") + 1]
        btch_name = split_path[split_path.index("experiment_0") - 1]
        series = pd.Series({"batch": btch_name, "checkpoint": chkp_name}).append(
            df.iloc[:, 1:].median(numeric_only=True)
        )
        summaries_list.append(series)
    summarized_df = pd.DataFrame(summaries_list)
    summarized_df["mean"] = summarized_df.loc[:, summarized_df.columns != "falls"].mean(axis=1)
    summarized_df.to_csv(f"{batch_path}/batch_summary.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", help="Configuration file")
    args = parser.parse_args()
    yaml_path_split = args.yaml.split("/")
    config_path = "/".join(yaml_path_split[:-1])
    config_name = yaml_path_split[-1][:-5]
    with initialize(config_path=config_path, job_name="mentor_app"):
        yaml_conf = compose(config_name=config_name, overrides=["hydra.run.dir=/tmp"])
        # Struct to normal :)
        yaml_conf = OmegaConf.to_container(yaml_conf)
        yaml_conf = OmegaConf.create(yaml_conf)
    default_train = yaml_conf["default_yaml_paths"]["train"]
    default_eval = yaml_conf["default_yaml_paths"]["eval"]
    default_eval_visual = yaml_conf["default_yaml_paths"]["eval_visual"]
    base_name = yaml_conf["logging_dirs"]["name"]
    base_output_dir = yaml_conf["logging_dirs"]["output_dir"]
    username = os.environ["USER"]
    all_combinations = generate_combinations()
    batch_experiment_details = pd.DataFrame()
    batch_path = os.path.join(base_output_dir, username, "run", base_name)
    current_experiment = 0
    if yaml_conf["train_params"]["train"]:
        assert not os.path.isdir(
            batch_path
        ), "Batch name already exists, change it if you want to train"
        os.mkdir(batch_path)
    if yaml_conf["general"]["mode"] == "grid":
        for combination in all_combinations:
            final_path, opts, experiment_details = generate_opts(combination)
            batch_experiment_details = pd.concat(
                (batch_experiment_details, experiment_details), axis=1
            )
            if yaml_conf["train_params"]["train"]:
                print(f"Starting training experiment: \n{experiment_details}")
                batch_experiment_details.to_csv(f"{batch_path}/batch_details.csv")
                p = subprocess.run(["python", "run.py", "--yaml", default_train, *opts])
            if yaml_conf["eval_params"]["eval"]:
                checkpoints = get_checkpoints(final_path)
                if checkpoints != None:
                    for checkpoint in checkpoints:
                        path = f"{final_path}/experiment_0/{checkpoint}.pth.tar"
                        print(
                            f"Starting eval experiment: \n{experiment_details}\ncheckpoint: {checkpoint}"
                        )
                        assert os.path.isdir(
                            f"{final_path}/experiment_0"
                        ), f'Eval path: {final_path}/experiment_0" does not exist '
                        p = subprocess.run(
                            [
                                "python",
                                "run.py",
                                "--yaml",
                                default_train,
                                "--play",
                                f"play={default_eval}",
                                f"experiment.resume.path={path}",
                                "environment.isaac.headless=True",
                            ]
                        )
                        check_stats()
        current_experiment += 1

    elif yaml_conf["general"]["mode"] == "random":
        num_random_iters = yaml_conf["train_params"]["num_random_runs"]
        random_idx = np.random.choice(len(all_combinations), num_random_iters, replace=False)
        for random_opt in random_idx:
            final_path, opts, experiment_details = generate_opts(all_combinations[random_opt])
            batch_experiment_details = pd.concat(
                (batch_experiment_details, experiment_details), axis=1
            )
            if yaml_conf["train_params"]["train"]:
                batch_experiment_details.to_csv(f"{batch_path}/batch_details.csv")
                print(f"Starting training experiment: \n{experiment_details}")
                p = subprocess.run(["python", "run.py", "--yaml", default_train, *opts])
            if yaml_conf["eval_params"]["eval"]:
                checkpoints = get_checkpoints(final_path)
                if checkpoints != None:
                    for checkpoint in checkpoints:
                        path = f"{final_path}/experiment_0/{checkpoint}.pth.tar"
                        print(
                            f"Starting eval experiment: \n{experiment_details}\ncheckpoint: {checkpoint}"
                        )
                        assert os.path.isdir(
                            f"{final_path}/experiment_0"
                        ), f'Eval path: {final_path}/experiment_0" does not exist '
                        p = subprocess.run(
                            [
                                "python",
                                "run.py",
                                "--yaml",
                                default_train,
                                "--play",
                                f"play={default_eval}",
                                f"play.experiment.resume.path={path}",
                                "environment.isaac.headless=True",
                            ]
                        )

                        # p = subprocess.run(
                        #     [
                        #         "python",
                        #         "run.py",
                        #         "--yaml",
                        #         default_train,
                        #         "--play",
                        #         f"play={default_eval_visual}",
                        #         f"play.experiment.resume.path={path}",
                        #         "environment.isaac.headless=False",
                        #     ]
                        # )
                        check_stats()
            current_experiment += 1
