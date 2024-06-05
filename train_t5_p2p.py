from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config, Trainer, TrainingArguments
from tqdm import tqdm
import numpy as np
import string
import os
import torch
from torch.utils.data import Dataset
import re
import random
from textgrids import TextGrid
from copy import deepcopy
from torch.nn.utils.rnn import pad_sequence

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
            # print(item)
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
            # print(interval)
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

        input_ids = [token_to_index[ph] for ph in phonemes if ph in token_to_index]
        labels = deepcopy(input_ids)
        
        return {'sentence': words, 'phonemes': phonemes, 'input_ids': input_ids, 'labels': labels}

splits = ['train', 'validation']

loaded_datasets = {
    'train': LibriSpeechAlignmentDataset([
        '/mnt/scratch/kudrinsk/eval_challenge/librispeech_alignments/train-clean-100/',
        '/mnt/scratch/kudrinsk/eval_challenge/librispeech_alignments/train-clean-360/',
        '/mnt/scratch/kudrinsk/eval_challenge/librispeech_alignments/dev-clean/',
    ]),
    'validation': LibriSpeechAlignmentDataset([
        '/mnt/scratch/kudrinsk/eval_challenge/librispeech_alignments/test-clean/',
    ]),
}

def get_phoneme_list():
    phonemes_list = [
        'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 
        'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 
        'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH', 'spn'
    ]
    return phonemes_list

tokens = ['<PAD>', '<EOS>', '<START_DEC>', '<UNK>', ' '] + get_phoneme_list()
token_set = set(tokens)

print(tokens)
token_to_index = {token: idx for idx, token in enumerate(tokens)}
index_to_token = {idx: token for token, idx in token_to_index.items()}
phoneme_indexes = [token_to_index[t] for t in get_phoneme_list()]

config = T5Config(
    vocab_size=len(tokens),
    num_layers=6,
    pad_token_id=token_to_index['<PAD>'], 
    eos_token_id=token_to_index['<EOS>'],
    decoder_start_token_id=token_to_index['<START_DEC>'],
)
model = T5ForConditionalGeneration(config).cuda()

run_name = 'phoneme_lm_gpt2'

training_args = TrainingArguments(
    output_dir=f"./{run_name}",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-6,
    per_device_train_batch_size=16,
    num_train_epochs=1000,
    weight_decay=0.01,
    dataloader_num_workers=32,
    run_name=run_name,
)

class CustomDataCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        input_ids = [torch.tensor(feature['input_ids']) for feature in features]
        labels = [torch.tensor(feature['labels']) for feature in features]

        # Pad input_ids and labels
        padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        # Create attention masks for the inputs
        attention_mask = (padded_input_ids != self.pad_token_id).long()

        return {
            'input_ids': padded_input_ids,
            'attention_mask': attention_mask,
            'labels': padded_labels
        }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=loaded_datasets['train'],
    eval_dataset=loaded_datasets['validation'],
    data_collator=CustomDataCollator(token_to_index['<PAD>']),
    # compute_metrics=compute_metrics,
)

trainer.train()