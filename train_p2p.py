from transformers import GPT2LMHeadModel, GPT2Config, Trainer, TrainingArguments
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

splits = ['train', 'validation']

loaded_datasets = {
    'train': concatenate_datasets([
        load_from_disk('./wikitext_full-phonemes-train-0.0-0.015'),
        load_from_disk('./wikitext_full-phonemes-train-0.015-0.03'),
        load_from_disk('./wikitext_full-phonemes-train-0.03-0.045'),
        load_from_disk('./wikitext_full-phonemes-train-0.045-0.06'),
    ]),
    'validation': concatenate_datasets([
        load_from_disk('./wikitext_full-phonemes-validation-0.0-0.015'),
        load_from_disk('./wikitext_full-phonemes-validation-0.015-0.03'),
        load_from_disk('./wikitext_full-phonemes-validation-0.03-0.045'),
        load_from_disk('./wikitext_full-phonemes-validation-0.045-0.06'),
    ])
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

# config = GPT2Config(
#     vocab_size=len(tokens),
#     n_positions=1024,
#     n_ctx=1024,
#     n_embd=768,
#     n_layer=6,
#     n_head=6,
#     pad_token_id=token_to_index['<PAD>'], 
#     eos_token_id=token_to_index['<EOS>'],
# )
# model = GPT2LMHeadModel(config).cuda()

# model = GPT2LMHeadModel.from_pretrained('./phoneme_lm_gpt2/checkpoint-67500/')

config = GPT2Config.from_pretrained('gpt2')

config.vocab_size = len(tokens)
config.pad_token_id = token_to_index['<PAD>']
config.eos_token_id = token_to_index['<EOS>']
model = GPT2LMHeadModel.from_pretrained('gpt2', config=config, ignore_mismatched_sizes=True)

run_name = 'phoneme_lm_gpt2_pretr'

training_args = TrainingArguments(
    output_dir=f"./{run_name}",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
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
