import torch
import torch.nn.functional as F
import pickle
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import evaluate
from torch.nn.utils.rnn import pad_sequence
import wandb
from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperConfig

from dataset import TextDataset, split_dataset

# Load the dataset
with open("/mnt/scratch/kudrinsk/eval_challenge/dataset_phonemes.pickle", "rb") as handle:
    data = pickle.load(handle)

datasets = split_dataset(data, ('train', 'test'))
train_ds = TextDataset(datasets['train'])
test_ds = TextDataset(datasets['test'])

# Initialize the processor
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")

# Config and model
config = WhisperConfig.from_pretrained("openai/whisper-tiny.en")
config.num_mel_bins = 256
model = WhisperForConditionalGeneration(config)

# Freeze all layers initially
# for param in model.parameters():
#     param.requires_grad = False

# # Unfreeze specific layers of the decoder (e.g., the last two layers)
# for param in model.model.decoder.layers[-2:].parameters():
#     param.requires_grad = True

# for param in model.proj_out.parameters():
#     param.requires_grad = True

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def _create_attention_mask_from_lengths(self, lengths, max_len=None):
        if max_len is None:
            max_len = max(lengths)
        batch_size = len(lengths)
        mask = torch.zeros(batch_size, max_len)
        for i, length in enumerate(lengths):
            mask[i, :length] = 1
        return mask.bool()

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        batch = {}
        
        input_features = [torch.tensor(feature[0]) for feature in features]
        input_features = pad_sequence(input_features, batch_first=True, padding_value=processor.tokenizer.pad_token_id)
        input_features = input_features.permute(0, 2, 1)
        
        padding = (0, 3000 - input_features.shape[-1])
        input_features = F.pad(input_features, padding)
        batch['input_features'] = input_features
        
        input_lens = torch.tensor([len(feature[0]) for feature in features])
        input_attention = self._create_attention_mask_from_lengths(input_lens)
        batch['attention_mask'] = input_attention

        label_features = [feature[1] for feature in features]
        label_features = self.processor(text=label_features)
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

# Initialize the data collator
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

# Initialize the WER metric
metric = evaluate.load("wer")

# Define the compute_metrics function
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

exp_name = 'whisper-tiny-unfrozen'
wandb.init(project="huggingface", name=exp_name)  # Update with your project and experiment name
# Define training arguments

wandb.save(os.path.abspath(__file__))
training_args = Seq2SeqTrainingArguments(
    output_dir=f"./{exp_name}/",  # change to a repo name of your choice
    per_device_train_batch_size=64,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-6,
    warmup_steps=500,
    max_steps=40000,
    # gradient_checkpointing=True,
    # fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["wandb"],
    # load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    # push_to_hub=True,
)

# Initialize the trainer
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Start the training
trainer.train()
