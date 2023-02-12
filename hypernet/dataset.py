from torch.utils.data import Dataset
import torch
import glob

class PoliciesDataset(Dataset):
    def __init__(self, path):
        """
        """
        super().__init__()
        self.policies_paths = glob.glob('/home/tom/PycharmProjects/IsaacGymEnvs/isaacgymenvs/runs/with_normalization/'
                                        '0*//**/H*.pth',
                                        recursive=True)
        self.num_policies = len(self.policies_paths)
        policy = torch.load(self.policies_paths[0])['model']
        actor_weights = [value for key, value in policy.items() if 'actor' in key]
        self.layer_shapes = [layer.shape for layer in actor_weights]

    def __getitem__(self, index):
        policy = torch.load(self.policies_paths[index])['model']
        actor_weights = [value for key, value in policy.items() if 'actor' in key]
        # load the checkpoint of the policy
        return actor_weights

    def __len__(self):
        return self.num_policies
