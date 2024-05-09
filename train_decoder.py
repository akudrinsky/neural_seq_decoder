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

from dataset import DecoderDataset, split_dataset
from models import GRUEncoder, TransformerSequenceEmbedding
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

from transformers import AutoTokenizer, GPT2LMHeadModel

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").cuda()

# Freeze all parameters in the model
# for param in model.parameters():
#     param.requires_grad = False
# # Unfreeze the parameters of the first transformer layer (h[0])
# for param in model.transformer.h[0].parameters():
#     param.requires_grad = True

print(model)

exp_name = 'gpt2_from_embeds_all_layers'

wandb.init(project='decoder', name=exp_name)

device = 'cuda'
epochs = 1000
batch_size = 400

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
    text_ids['input_ids'].to(device)

    return X_padded.to(device), X_mask.to(device), gt_embeds.to(device), text_ids['input_ids'].to(device), text_ids['attention_mask'].to(device), transcriptions

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
optimizer = AdamW(model.parameters(), lr=5e-6)
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

    for neuro, neuro_mask, gt_embeds, text_ids, text_mask, gt_texts in tqdm(train_loader, desc=f'Train {epoch}'):
        optimizer.zero_grad()
        text_embeds = model.transformer.wte.weight[text_ids, :]

        input_embeds = torch.cat((gt_embeds.unsqueeze(1), text_embeds), dim=1)
        input_mask = torch.cat((torch.ones(text_ids.shape[0], 1).to(device), text_mask), dim=1)

        output_ids = torch.cat((torch.tensor([-100] * text_ids.shape[0]).cuda()[:, None], text_ids), dim=1) 
        # {'bos_token': '<|endoftext|>',
        # 'eos_token': '<|endoftext|>',
        # 'unk_token': '<|endoftext|>'}
        # all are 50256

        outputs = model(inputs_embeds=input_embeds, attention_mask=input_mask, labels=output_ids)

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
        for neuro, neuro_mask, gt_embeds, text_ids, text_mask, gt_texts in tqdm(eval_loader, desc=f'Eval {epoch}'):
            text_embeds = model.transformer.wte.weight[text_ids, :]

            input_embeds = torch.cat((gt_embeds.unsqueeze(1), text_embeds), dim=1)
            input_mask = torch.cat((torch.ones(text_ids.shape[0], 1).to(device), text_mask), dim=1)
    
            output_ids = torch.cat((text_ids, torch.tensor([-100] * text_ids.shape[0]).cuda()[:, None]), dim=1)
    
            outputs = model(inputs_embeds=input_embeds, attention_mask=input_mask, labels=output_ids)
    
            loss = outputs.loss
            total_eval_loss += loss.item()

            predicted_token_ids = outputs.logits.argmax(-1)
            pred_texts = tokenizer.batch_decode(predicted_token_ids, skip_special_tokens=True)
            print(f'GT: {gt_texts[0]}\nPRED: {pred_texts[0]}')

            eval_wer += compute_wer(pred_texts, gt_texts)

    avg_eval_loss = total_eval_loss / len(eval_loader)
    eval_wer = eval_wer / len(eval_loader)
    
    print(f"Epoch {epoch+1}, Average Evaluation Loss: {avg_eval_loss}, WER: {eval_wer}")
    wandb.log({"eval/loss": avg_eval_loss, "eval/wer": eval_wer})

    torch.save(model.state_dict(), f'{save_dir}/{epoch}.pth')
