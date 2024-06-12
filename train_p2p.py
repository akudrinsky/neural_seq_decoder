from transformers import Trainer, TrainingArguments

from data import LibriSpeechAlignmentDataset
from models import get_phoneme_gpt2_model
from custom_tokenizers import PhonemeTokenizer
from data_collators import P2PDataCollator

import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--run_name', type=str, default='phoneme_lm_gpt2_newcode')

args = arg_parser.parse_args()

run_name = args.run_name

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

tokenizer = PhonemeTokenizer()

model = get_phoneme_gpt2_model(vocab_size=tokenizer.token_list,
                               pad_token_id=tokenizer.pad_token,
                               eos_token_id=tokenizer.eos_token)

data_collator = P2PDataCollator(pad_token_id=tokenizer.pad_token)

training_args = TrainingArguments(
    output_dir=f"./checkpoints/{run_name}",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    num_train_epochs=1000,
    weight_decay=0.01,
    dataloader_num_workers=32,
    run_name=run_name,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=loaded_datasets['train'],
    eval_dataset=loaded_datasets['validation'],
    data_collator=data_collator,
)

trainer.train()
