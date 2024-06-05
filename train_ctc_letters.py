import os
import wandb
import pickle
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import jiwer

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from edit_distance import SequenceMatcher

from dataset import DecoderDataset, split_dataset
from models import GRUEncoder
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

from itertools import groupby
import string

def remove_consecutive_duplicates(phoneme_list):
    return [key for key, group in groupby(phoneme_list)]

phoneme_list = list('_' + string.ascii_lowercase + ' ')
id2ph = {i: ph for i, ph in enumerate(phoneme_list)}
ph2id = {ph: i for i, ph in enumerate(phoneme_list)}

# Display the list
print(phoneme_list)

model = GRUEncoder(output_embed_dim=len(phoneme_list), layer_dim=10)

print(model)

exp_name = 'ctc_characters'

wandb.init(project='ctc', name=exp_name)

device = 'cuda'
epochs = 1000
batch_size = 128

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

    # print(transcriptions)

    transcriptions = [''.join([ch for ch in s.lower() if ch in phoneme_list]) for s in transcriptions]
    transcription_ids = [torch.tensor([ph2id[p] for p in sent.lower()]) for sent in transcriptions]
    transcription_lens = torch.tensor([len(t) for t in transcription_ids], device=device)
    
    transcription_ids = pad_sequence(transcription_ids, batch_first=True, padding_value=0).to(device)
    
    return X_padded.to(device), X_mask.to(device), transcription_ids, transcription_lens, transcriptions

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
optimizer = AdamW(model.parameters(), lr=0.02, betas=(0.9, 0.999), eps=0.1, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=10000)
loss_ctc = torch.nn.CTCLoss(blank=ph2id['_'], reduction="mean", zero_infinity=True)


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
    train_cer = 0
    total_edit_distance = 0
    total_seq_length = 0

    for neuro, neuro_mask, text_ids, text_lens, gt_texts in tqdm(train_loader, desc=f'Train {epoch}'):
        optimizer.zero_grad()

        neuro += torch.randn(neuro.shape, device=device) * 0.8
        neuro += (torch.randn([neuro.shape[0], 1, neuro.shape[2]], device=device) * 0.2)

        pred = model(neuro)

        loss = loss_ctc(
            torch.permute(pred.log_softmax(2), [1, 0, 2]),
            text_ids,
            ((neuro_mask.sum(dim=1) - model.kernelLen) / model.strideLen).to(torch.int32),
            text_lens,
        )
        
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

        predicted_token_ids = pred.argmax(-1)
        seqs = [[id2ph[ph.item()] for ph in predicted_token_ids[i] if id2ph[ph.item()] != '_'] for i in range(len(predicted_token_ids))]
        pred_texts = [''.join(remove_consecutive_duplicates(seq)) for seq in seqs]
        
        train_wer += compute_wer(pred_texts, gt_texts)

        total_train_loss += loss.item()

        for gt, pred in zip(gt_texts, pred_texts):
            matcher = SequenceMatcher(
                a=gt, b=pred
            )
            total_edit_distance += matcher.distance()
            total_seq_length += len(gt)

    train_cer = total_edit_distance / total_seq_length
    avg_train_loss = total_train_loss / len(train_loader)
    train_wer = train_wer / len(train_loader)
    
    print(f"Epoch {epoch+1}, Average Training Loss: {avg_train_loss}, WER: {train_wer}, CER: {train_cer}")
    wandb.log({"train/loss": avg_train_loss, 'train/wer': train_wer, 'train/cer': train_cer})

    model.eval()
    total_eval_loss = 0
    eval_wer = 0
    eval_cer = 0
    total_edit_distance = 0
    total_seq_length = 0

    with torch.no_grad():
        for neuro, neuro_mask, text_ids, text_lens, gt_texts in tqdm(eval_loader, desc=f'Eval {epoch}'):
            pred = model(neuro)
    
            loss = loss_ctc(
                torch.permute(pred.log_softmax(2), [1, 0, 2]),
                text_ids,
                ((neuro_mask.sum(dim=1) - model.kernelLen) / model.strideLen).to(torch.int32),
                text_lens,
            )
            total_eval_loss += loss.item()

            predicted_token_ids = pred.argmax(-1)
            seqs = [[id2ph[ph.item()] for ph in predicted_token_ids[i] if id2ph[ph.item()] != '_'] for i in range(len(predicted_token_ids))]
            pred_texts = [''.join(remove_consecutive_duplicates(seq)) for seq in seqs]
            print(f'GT: {gt_texts[0]}\nPRED: {pred_texts[0]}')

            eval_wer += compute_wer(pred_texts, gt_texts)

            for gt, pred in zip(gt_texts, pred_texts):
                matcher = SequenceMatcher(
                    a=gt, b=pred
                )
                total_edit_distance += matcher.distance()
                total_seq_length += len(gt)

    eval_cer = total_edit_distance / total_seq_length
    avg_eval_loss = total_eval_loss / len(eval_loader)
    eval_wer = eval_wer / len(eval_loader)
    
    print(f"Epoch {epoch+1}, Average Evaluation Loss: {avg_eval_loss}, WER: {eval_wer}, CER: {eval_cer}")
    wandb.log({"eval/loss": avg_eval_loss, "eval/wer": eval_wer, 'eval/cer': eval_cer})

    torch.save(model.state_dict(), f'{save_dir}/{epoch}.pth')
