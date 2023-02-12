import argparse
import glob
import itertools as it
import os
import subprocess
import torch
import numpy as np
import pandas as pd
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
NUM_ITERS = 10000

if __name__ == "__main__":
    for run_idx in tqdm(range(NUM_ITERS)):
        weights = torch.rand(8)
        weights /= weights.sum()
        heading_w = weights[0]
        up_w = weights[1]
        progress_w = weights[2]
        actions_w = weights[3]
        energy_w = weights[4]
        limits_w = weights[5]
        death_w = weights[6]
        acc_w = weights[7]
        exp_name = str([round(w.item(), 2) for w in weights])
        exp_name = exp_name.replace(' ', '').replace('[', '').replace(']', '').replace('.', '_').replace(',', '-')
        # exp_name = 'ted'
        p = subprocess.run(
            [
                "python",
                "train.py",
                "task=HumanoidCustom",
                f"+full_experiment_name={exp_name}",
                f"+headingWeight={heading_w}",
                f"+upWeight={up_w}",
                f"+progressWeight={progress_w}",
                f"+actionsCost={actions_w}",
                f"+energyCost={energy_w}",
                f"+jointsAtLimitCost={limits_w}",
                f"+deathCost={death_w}",
                f"+accCost={acc_w}",
            ]
        )

        p = subprocess.run(
            [
                "python",
                "train.py",
                "task=HumanoidCustom",
                f"+full_experiment_name={exp_name}",
                f"+headingWeight={heading_w}",
                f"+upWeight={up_w}",
                f"+progressWeight={progress_w}",
                f"+actionsCost={actions_w}",
                f"+energyCost={energy_w}",
                f"+jointsAtLimitCost={limits_w}",
                f"+deathCost={death_w}",
                f"+accCost={acc_w}",
                "test=True",
                f"checkpoint=runs/{exp_name}/nn/HumanoidCustom.pth"
            ]
        )
