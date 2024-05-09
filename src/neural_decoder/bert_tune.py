import torch
import torch.nn as nn
import pickle
import jiwer
import wandb

from transformers import AutoTokenizer, BartForConditionalGeneration
from transformers.generation import GenerationConfig
from transformers import Trainer, TrainingArguments
from transformers.models.bart.modeling_bart import shift_tokens_right
from transformers import DataCollatorForSeq2Seq
from dataset import SpeechDataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import evaluate

class BART(nn.Module):
    def __init__(self):
        super(BART, self).__init__()
        self.bart = BartForConditionalGeneration.from_pretrained("facebook/bart-base", attention_dropout=0.1)
        self.linear = nn.Linear(256, 768)

        self.bart.config.output_attentions = True
        self.bart.config.output_hidden_states = True

    def forward(self, **kwargs):
        inputs_embeds = self.linear(kwargs['inputs_embeds'])
        kwargs['inputs_embeds'] = inputs_embeds
        return self.bart(**kwargs)

with open("/mnt/scratch/kudrinsk/eval_challenge/ptDecoder_ctc", "rb") as handle:
    loadedData = pickle.load(handle)

def create_attention_mask_from_lengths(lengths, max_len=None):
    if max_len is None:
        max_len = max(lengths)  # Calculate max length if not provided
    batch_size = len(lengths)
    mask = torch.zeros(batch_size, max_len)
    for i, length in enumerate(lengths):
        mask[i, :length] = 1
    return mask

def collate(batch):
    X, y, X_lens, y_lens, days, transcriptions = zip(*batch)

    X_padded = pad_sequence(X, batch_first=True, padding_value=0)
    X_attention_mask = create_attention_mask_from_lengths(X_lens, max_len=X_padded.shape[1])

    with tokenizer.as_target_tokenizer():
        target_encodings = tokenizer(transcriptions, max_length=256, truncation=False)

    target = pad_sequence([torch.tensor(t) for t in target_encodings["input_ids"]], batch_first=True, padding_value=0)

    return {"inputs_embeds": X_padded, 
           "attention_mask": X_attention_mask, 
           "labels": target}

train_ds = SpeechDataset(loadedData["train"], transform=None)
test_ds = SpeechDataset(loadedData["test"])

train_loader = DataLoader(
    train_ds,
    batch_size=16,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    collate_fn=collate,
)
eval_loader = DataLoader(
    test_ds,
    batch_size=16,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    collate_fn=collate,
)

model = BART()
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")

import torch
from torch.optim import AdamW
import wandb
from tqdm import tqdm

def train_and_evaluate(model, train_loader, eval_loader, tokenizer, compute_metrics, device='cuda', epochs=1000):
    # Move model to the specified device
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_loader, desc=f'Train {epoch}'):
            optimizer.zero_grad()
            inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
            labels = batch["labels"].to(device)

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Average Training Loss: {avg_train_loss}")
        wandb.log({"train/loss": avg_train_loss})

        # Evaluate the model
        model.eval()
        total_eval_loss = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc=f'Eval {epoch}'):
                inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
                labels = batch["labels"].to(device)

                outputs = model(**inputs, labels=labels)
                loss = outputs.loss
                total_eval_loss += loss.item()

                predictions = torch.argmax(outputs.logits, dim=-1)
                all_predictions.extend(tokenizer.batch_decode(predictions, skip_special_tokens=True))
                all_labels.extend(tokenizer.batch_decode(labels, skip_special_tokens=True))

        avg_eval_loss = total_eval_loss / len(eval_loader)
        print(f"Epoch {epoch+1}, Average Evaluation Loss: {avg_eval_loss}")
        wandb.log({"eval/loss": avg_eval_loss})

        # Compute the WER
        metrics = compute_metrics(all_predictions, all_labels)
        print(f"Epoch {epoch+1}, WER: {metrics['eval/wer']}")
        wandb.log(metrics)

def compute_metrics(predictions, references):
    normalizer = BasicTextNormalizer()
    normalized_predictions = [normalizer(pred) for pred in predictions]
    normalized_references = [normalizer(ref) for ref in references]
    wer = jiwer.wer(normalized_references, normalized_predictions)
    
    return {
        "eval/wer": 100 * wer
    }

wandb.init(project='BART')
train_and_evaluate(model, train_loader, eval_loader, tokenizer, compute_metrics)
