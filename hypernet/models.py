import torch
from collections import OrderedDict
from torch import nn

class HyperPolicy(torch.nn.Module):

    def __init__(self, input_size, fc_shapes):
        super(HyperPolicy, self).__init__()
        self.input_size = input_size
        self.output_net_shapes = fc_shapes
        self.flattened_output_shape = sum([shape[0] * shape[1] if len(shape) == 2 else shape[0] for shape in fc_shapes])
        self.model = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.input_size, 512)),
            ('elu1', nn.ELU()),
            ('fc2', nn.Linear(512, self.flattened_output_shape//4)),
            ('elu2', nn.ELU()),
            ('projection', nn.Linear(self.flattened_output_shape//4, self.flattened_output_shape))
            ]))

    def forward(self, reward_weighting):
        output = self.model(reward_weighting)

        return output

