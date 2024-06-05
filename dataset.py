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
        self._id2ph = [
            'EPS_CTC', 'AA', 'AE', 'AH', 'AO', 'AW',
            'AY', 'B',  'CH', 'D', 'DH',
            'EH', 'ER', 'EY', 'F', 'G',
            'HH', 'IH', 'IY', 'JH', 'K',
            'L', 'M', 'N', 'NG', 'OW',
            'OY', 'P', 'R', 'S', 'SH',
            'T', 'TH', 'UH', 'UW', 'V',
            'W', 'Y', 'Z', 'ZH', ' '
        ]
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        phonemes = [self._id2ph[p] for p in self.data[idx]['phonemes'] if p > 0]
        phonemes.append('<EOS>')
        return torch.tensor(self.data[idx]['neuro']).float(), phonemes
