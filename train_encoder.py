import os
import wandb
import pickle
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from dataset import EncoderDataset, split_dataset
from models import GRUEncoder, TransformerSequenceEmbedding

exp_name = 'gru_small_dropout'

wandb.init(project='encoder', name=exp_name)

device = 'cuda'
epochs = 1000
batch_size = 400

save_dir = f'../ckpts/{exp_name}/'
os.makedirs(save_dir, exist_ok=True)

with open("/mnt/scratch/kudrinsk/eval_challenge/gpt2_text_embeds.pickle", "rb") as handle:
    data = pickle.load(handle)

datasets = split_dataset(data, ('train', 'test'))

train_ds = EncoderDataset(datasets['train'])
test_ds = EncoderDataset(datasets['test'])

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
    X, embeds = zip(*batch)
    X_lens = [x.shape[0] for x in X]

    X_padded = pad_sequence(X, batch_first=True, padding_value=0)
    X_mask = create_attention_mask_from_lengths(X_lens, max_len=X_padded.shape[1])

    embeds = torch.stack(embeds)

    return X_padded.to(device), X_mask.to(device), embeds.to(device)

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

model = GRUEncoder()

model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)
loss_function = nn.L1Loss()

for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    for neuro, neuro_mask, embeds in tqdm(train_loader, desc=f'Train {epoch}'):
        optimizer.zero_grad()

        # neuro += torch.randn(neuro.shape, device=device) * 0.8
        # neuro += torch.randn([neuro.shape[0], 1, neuro.shape[2]], device=device) * 0.2

        outputs = model(neuro)

        loss = loss_function(outputs, embeds)
        
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
    
    avg_train_loss = total_train_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Average Training Loss: {avg_train_loss}")
    wandb.log({"train/loss": avg_train_loss})

    model.eval()
    total_eval_loss = 0

    with torch.no_grad():
        for neuro, neuro_mask, embeds in tqdm(eval_loader, desc=f'Eval {epoch}'):
            outputs = model(neuro)

            loss = loss_function(embeds, outputs)
            total_eval_loss += loss.item()

    avg_eval_loss = total_eval_loss / len(eval_loader)
    print(f"Epoch {epoch+1}, Average Evaluation Loss: {avg_eval_loss}")
    wandb.log({"eval/loss": avg_eval_loss})

    torch.save(model.state_dict(), f'{save_dir}/{epoch}.pth')
