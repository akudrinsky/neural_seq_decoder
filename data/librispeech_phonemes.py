import os
from torch.utils.data import Dataset
import re
from textgrids import TextGrid


class LibriSpeechAlignmentDataset(Dataset):
    def __init__(self, root_folders):
        self.files = []
        for root_folder in root_folders:
            for subdir, _, files in os.walk(root_folder):
                for file in files:
                    if file.endswith('.TextGrid'):
                        self.files.append(os.path.join(subdir, file))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        textgrid = TextGrid()
        textgrid.read(file_path)

        words = ''
        phonemes = []

        word_intervals = []
        phone_intervals = []

        for item in textgrid:
            if item == 'words':
                for interval in textgrid[item]:
                    word_intervals.append(interval)
            elif item == 'phones':
                for interval in textgrid[item]:
                    if interval.text == 'sil':
                        continue
                    phone_intervals.append(interval)
            else:
                print('UNK', item)

        for interval in word_intervals:
            if interval.text:
                words += interval.text + ' '
                # Find corresponding phone intervals
                for phone_interval in phone_intervals:
                    if phone_interval.xmax <= interval.xmax and phone_interval.xmin >= interval.xmin:
                        phoneme = re.sub(r'\d+', '', phone_interval.text)
                        phonemes.append(phoneme)
                # Add space token after each word's phonemes
                phonemes.append(' ')

        words = words.strip()  # Remove trailing space

        return {'sentence': words, 'phonemes': phonemes}