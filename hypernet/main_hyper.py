import torch
from dataset import PoliciesDataset
from models import HyperPolicy
INPUT_SHAPE = 7

if __name__ == '__main__':
    dataset = PoliciesDataset('isaacgymenvs/runs')
    model = HyperPolicy(INPUT_SHAPE, dataset.layer_shapes)
    pass