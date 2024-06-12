import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Config
from datasets import load_metric
import numpy as np
from tqdm import tqdm
import random
import string
import os
from torch.utils.data import Dataset
import re
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



# Define phoneme tokenizer logic
def get_phoneme_list():
    phonemes_list = [
        'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 
        'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 
        'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH', 'spn'
    ]
    return phonemes_list

tokens = ['<PAD>', '<EOS>', '<START_DEC>', '<UNK>', ' '] + get_phoneme_list()
token_set = set(tokens)

token_to_index = {token: idx for idx, token in enumerate(tokens)}
index_to_token = {idx: token for token, idx in token_to_index.items()}
phoneme_indexes = [token_to_index[t] for t in get_phoneme_list()]

# Function to convert ids to tokens
def decode_ids(ids):
    return ''.join([index_to_token[id] for id in ids if id in index_to_token])

# Function to encode tokens to ids
def encode_tokens(tokens):
    return [token_to_index[token] for token in tokens if token in token_to_index]

# Load the model
config = GPT2Config.from_pretrained('gpt2')
config.vocab_size = len(tokens)
config.pad_token_id = token_to_index['<PAD>']
config.eos_token_id = token_to_index['<EOS>']

model = GPT2LMHeadModel.from_pretrained('./phoneme_lm_gpt2_pretr/checkpoint-330000/', config=config).cuda()

# Load the datasets
train_dataset = loaded_datasets['train']
val_dataset = loaded_datasets['validation']

# Define evaluation metrics
accuracy_metric = load_metric("accuracy")

# DataLoader for validation set
val_dataloader = DataLoader(val_dataset, batch_size=8, collate_fn=CustomDataCollator(token_to_index['<PAD>']), shuffle=False)

def evaluate(model, dataloader, accuracy_metric):
    model.eval()
    total_loss = 0
    num_batches = 0
    all_predictions = []
    all_labels = []

    for batch in tqdm(dataloader):
        inputs = batch['input_ids'].cuda()
        labels = batch['labels'].cuda()
        attention_mask = batch['attention_mask'].cuda()

        with torch.no_grad():
            outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

        total_loss += loss.item()
        num_batches += 1

        predictions = torch.argmax(logits, dim=-1)
        all_predictions.extend(predictions.cpu().numpy().flatten())
        all_labels.extend(labels.cpu().numpy().flatten())

    accuracy = accuracy_metric.compute(predictions=all_predictions, references=all_labels)
    avg_loss = total_loss / num_batches
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return avg_loss, accuracy, perplexity

# Evaluate on validation set
# val_loss, val_accuracy, val_perplexity = evaluate(model, val_dataloader, accuracy_metric)
# print(f"Validation Loss: {val_loss}")
# print(f"Validation Accuracy: {val_accuracy['accuracy']}")
# print(f"Validation Perplexity: {val_perplexity}")

# Function to generate sample continuations
def generate_samples(model, dataset, num_samples=5):
    model.eval()
    samples = [i for i in range(num_samples)]
    for idx in samples:
        sample = dataset[idx]
        sentence = sample['sentence']
        phonemes = sample['phonemes']
        input_ids = sample['input_ids']

        # Get 50% length prefix from sample
        prefix_length = len(input_ids) // 2
        prefix_ids = input_ids[:prefix_length]

        prefix = decode_ids(prefix_ids)
        correct_continuation = decode_ids(input_ids[prefix_length:])

        print(f"Full sentence: {sentence}")
        print(f"Prefix: {prefix}")
        print(f"Correct Continuation: {correct_continuation}")

        # Generate 3 continuations
        continuations = []
        for _ in range(1):
            input_tensor = torch.tensor([prefix_ids]).cuda()
            mask = torch.ones_like(input_tensor)
            output = model.generate(input_tensor, attention_mask=mask, max_length=len(input_ids) + 10, num_return_sequences=1, temperature=0.5, do_sample=True)
            continuation = decode_ids(output[0].cpu().numpy())
            if '<EOS>' in continuation:
                continuation = continuation[:continuation.find('<EOS>')]
            else:
                print('NO EOS!!')
            continuations.append(continuation[len(prefix):])

        for i, continuation in enumerate(continuations):
            print(f"Sampled Continuation {i+1}: {continuation}")
        print()

# Generate samples from validation set
generate_samples(model, val_dataset)
