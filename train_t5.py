import os
import wandb
import pickle
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import torch.nn.functional as F
import jiwer

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from dataset import DecoderDataset, split_dataset
from models import GRUEncoder, TransformerSequenceEmbedding
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

from transformers import AutoTokenizer, T5ForConditionalGeneration, T5Config

class NeuroT5(nn.Module):
    def __init__(self):
        super().__init__()

        self.unfolder = torch.nn.Unfold(
            (32, 1), dilation=1, padding=0, stride=4
        )
        self.linear = nn.Linear(8192, 512)

        self.t5 = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

    def forward(self, neuro, neuro_mask, labels=None):
        # print(neuro.shape)
        neuro = self.unfolder(neuro.transpose(1, 2).unsqueeze(3)).transpose(1, 2)
        # print(neuro.shape)
        stride_length = 4
        # print(neuro_mask.shape)
        neuro_mask = neuro_mask[:, ::stride_length][:, :neuro.shape[1]]
        # print(neuro_mask.shape)
        
        input_embeds = self.linear(neuro)

        outputs = self.t5(inputs_embeds=input_embeds, attention_mask=neuro_mask, labels=labels)
        return outputs

model = NeuroT5()
tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")

exp_name = 'neuroT5_unfolder'

wandb.init(project='decoder', name=exp_name)

device = 'cuda'
epochs = 1000
batch_size = 240

save_dir = f'../ckpts/{exp_name}/'
os.makedirs(save_dir, exist_ok=True)

with open("/mnt/scratch/kudrinsk/eval_challenge/gpt2_text_embeds.pickle", "rb") as handle:
    data = pickle.load(handle)

datasets = split_dataset(data, ('train', 'test'))

train_ds = DecoderDataset(datasets['train'])
test_ds = DecoderDataset(datasets['test'])

print('Dataset lengths: ', len(train_ds), len(test_ds))

def create_attention_mask_from_lengths(lengths, max_len=None):
    if max_len is None:
        max_len = max(lengths)  # Calculate max length if not provided
    batch_size = len(lengths)
    mask = torch.zeros(batch_size, max_len)
    for i, length in enumerate(lengths):
        mask[i, :length] = 1
    return mask.bool()

def collate(batch):
    X, gt_embeds, transcriptions = zip(*batch)
    X_lens = [x.shape[0] for x in X]

    X_padded = pad_sequence(X, batch_first=True, padding_value=0)
    X_mask = create_attention_mask_from_lengths(X_lens, max_len=X_padded.shape[1])

    gt_embeds = torch.stack(gt_embeds)

    # print(transcriptions)
    text_ids = tokenizer(transcriptions, return_tensors='pt', padding=True)
    text_ids['input_ids'][~text_ids['attention_mask']] = -100

    return X_padded.to(device), X_mask.to(device), text_ids['input_ids'].to(device), text_ids['attention_mask'].to(device), transcriptions

train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    # pin_memory=True,
    collate_fn=collate,
)
eval_loader = DataLoader(
    test_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    # pin_memory=True,
    collate_fn=collate,
)

model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-3)
loss_function = nn.L1Loss()


def compute_wer(predictions, references):
    normalizer = BasicTextNormalizer()
    normalized_predictions = [normalizer(pred) for pred in predictions]
    normalized_references = [normalizer(ref) for ref in references]
    wer = jiwer.wer(normalized_references, normalized_predictions)
    return 100 * wer


for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    train_wer = 0

    for neuro, neuro_mask, text_ids, text_mask, gt_texts in tqdm(train_loader, desc=f'Train {epoch}'):
        optimizer.zero_grad()

        outputs = model(neuro, neuro_mask, labels=text_ids)

        loss = outputs.loss
        
        loss.backward()
        optimizer.step()

        predicted_token_ids = outputs.logits.argmax(-1)
        pred_texts = tokenizer.batch_decode(predicted_token_ids, skip_special_tokens=True)

        train_wer += compute_wer(pred_texts, gt_texts)

        total_train_loss += loss.item()
    
    avg_train_loss = total_train_loss / len(train_loader)
    train_wer = train_wer / len(train_loader)
    
    print(f"Epoch {epoch+1}, Average Training Loss: {avg_train_loss}, WER: {train_wer}")
    wandb.log({"train/loss": avg_train_loss, 'train/wer': train_wer})

    model.eval()
    total_eval_loss = 0
    eval_wer = 0

    with torch.no_grad():
        for neuro, neuro_mask, text_ids, text_mask, gt_texts in tqdm(eval_loader, desc=f'Eval {epoch}'):
            outputs = model(neuro, neuro_mask, labels=text_ids)
    
            loss = outputs.loss
            total_eval_loss += loss.item()

            predicted_token_ids = outputs.logits.argmax(-1)
            pred_texts = tokenizer.batch_decode(predicted_token_ids, skip_special_tokens=True)
            if eval_wer == 0: # first batch
                print(f'GT: {gt_texts[0]}\nPRED: {pred_texts[0]}')
                print(f'GT: {gt_texts[1]}\nPRED: {pred_texts[1]}')
                print(f'GT: {gt_texts[2]}\nPRED: {pred_texts[2]}')

            eval_wer += compute_wer(pred_texts, gt_texts)

    avg_eval_loss = total_eval_loss / len(eval_loader)
    eval_wer = eval_wer / len(eval_loader)
    
    print(f"Epoch {epoch+1}, Average Evaluation Loss: {avg_eval_loss}, WER: {eval_wer}")
    wandb.log({"eval/loss": avg_eval_loss, "eval/wer": eval_wer})

    torch.save(model.state_dict(), f'{save_dir}/{epoch}.pth')
