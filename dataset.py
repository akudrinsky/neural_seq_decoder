import torch
from torch.utils.data import Dataset

def split_dataset(data, keys=('train', 'test')):
    datasets = {key: [] for key in keys}
    for sample in data:
        if sample['split_name'] not in keys:
            continue
        datasets[sample['split_name']].append(sample)
    return datasets

class EncoderDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]['neuro']).float(), self.data[idx]['embedding'].float()


class DecoderDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]['neuro']).float(), self.data[idx]['embedding'].float(), self.data[idx]['text']
