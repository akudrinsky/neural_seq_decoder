from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from tqdm import tqdm
from datasets import load_from_disk
import string
from speechbrain.inference.text import GraphemeToPhoneme

splits = ['train', 'validation']

loaded_datasets = {}
for split in splits:
    dataset_path = f"./wikitext-processed-{split}"
    loaded_datasets[split] = load_from_disk(dataset_path)

def get_phoneme_list():
    g2p = GraphemeToPhoneme.from_hparams("speechbrain/soundchoice-g2p", savedir="pretrained_models/soundchoice-g2p")
    return g2p.phonemes

tokens = ['<PAD>', '<EOS>', '<START_DEC>', '<UNK>', ' '] + list(string.ascii_lowercase) + get_phoneme_list()
token_set = set(tokens)

print(tokens)
token_to_index = {token: idx for idx, token in enumerate(tokens)}
index_to_token = {idx: token for token, idx in token_to_index.items()}

from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config

config = T5Config(
    vocab_size=len(tokens),
    num_layers=6,
    pad_token_id=token_to_index['<PAD>'], 
    eos_token_id=token_to_index['<EOS>'],
    decoder_start_token_id=token_to_index['<START_DEC>'],
)
model = T5ForConditionalGeneration(config).cuda()

from datasets import Dataset
from transformers import Trainer, TrainingArguments

def valid_tokens(sample):
    for char in sample['sentences']:
        if char not in token_set:
            print(char, sample)
            return False
    return True

for split in splits:
    loaded_datasets[split] = loaded_datasets[split].filter(valid_tokens)

def encode_examples(sample):
    model_inputs = {}
    model_inputs['input_ids'] = [token_to_index[ph] for ph in sample['phonemes']]
    model_inputs['labels'] = [token_to_index[c] for c in sample['sentences']]
    return model_inputs

for split in splits:
    loaded_datasets[split] = loaded_datasets[split].map(encode_examples)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=48,
    num_train_epochs=1000,
    weight_decay=0.01,
    dataloader_num_workers=32,
)

import torch
from torch.nn.utils.rnn import pad_sequence

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
)

trainer.train()