import argparse
import glob
import itertools as it
import os
import subprocess

import numpy as np
import pandas as pd
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf


if __name__ == "__main__":
    p = subprocess.run(
        [
            "python",
            "train.py",
            "task=HumanoidCustom"
        ]
    )
