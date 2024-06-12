import os
import wandb
import pickle
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from edit_distance import SequenceMatcher
from dataset import DecoderDataset, split_dataset
from transformers import GPT2LMHeadModel, GPT2Config, BertModel, BertConfig
from transformers import get_linear_schedule_with_warmup


USE_WANDB = False

# Define the phoneme list and mappings

def get_phoneme_list():
    phonemes_list = [
        'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY',
        'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P',
        'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH', 'spn'
    ]
    return phonemes_list

tokens = ['<PAD>', '<EOS>', '<START_DEC>', '<UNK>', ' '] + get_phoneme_list()
print(f'Number of tokens: {len(tokens)}')
id2ph = {i: ph for i, ph in enumerate(tokens)}
ph2id = {ph: i for i, ph in enumerate(tokens)}

print(tokens)

# Define the BERT-based Encoder
class BERTEncoder(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=768, layer_dim=4):
        super(BERTEncoder, self).__init__()

        self.linear = nn.Linear(input_dim, hidden_dim)

        config = BertConfig(
            hidden_size=hidden_dim,
            num_hidden_layers=layer_dim,
            num_attention_heads=12,
            intermediate_size=hidden_dim*4,
            # hidden_dropout_prob=0.1,
            # attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
        )
        self.bert = BertModel(config)

    def forward(self, x):
        x = self.linear(x)
        return self.bert(inputs_embeds=x).last_hidden_state

# Define the model class with encoder and decoder
class Seq2SeqModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2SeqModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, neuro, neuro_mask, text_ids_x, text_ids, text_mask):
        encoder_outputs = self.encoder(neuro)
        decoder_outputs = self.decoder(input_ids=text_ids_x, attention_mask=text_mask, encoder_hidden_states=encoder_outputs, encoder_attention_mask=neuro_mask, labels=text_ids)
        return decoder_outputs

# Initialize the BERT Encoder
encoder = BERTEncoder(input_dim=1024, hidden_dim=768, layer_dim=2)

# Load the GPT-2 model with cross-attention enabled
config = GPT2Config.from_pretrained('gpt2')
config.vocab_size = len(tokens)
config.add_cross_attention = True
config.pad_token_id = ph2id['<PAD>']
config.eos_token_id = ph2id['<EOS>']

decoder = GPT2LMHeadModel.from_pretrained('./phoneme_lm_gpt2_pretr/checkpoint-330000/', config=config)

# for name, param in decoder.named_parameters():
#     if 'crossattention' not in name or 'cross_attn' not in name:
#         param.requires_grad = False

print(encoder)
print(decoder)

# Create the Seq2Seq model
model = Seq2SeqModel(encoder, decoder)
model.load_state_dict(torch.load('../ckpts/badr_code/16.pth')['model_state_dict'])

# Experiment settings
exp_name = 'badr_code_check_metrics'

if USE_WANDB:
    wandb.init(project='ctc', name=exp_name)

device = 'cuda'
epochs = 1000
batch_size = 128

save_dir = f'../ckpts/{exp_name}/'
os.makedirs(save_dir, exist_ok=True)

# Load the dataset
with open("/mnt/scratch/kudrinsk/eval_challenge/dataset_phonemes.pickle", "rb") as handle:
    data = pickle.load(handle)
print(data[0])

datasets = split_dataset(data, ('train', 'test'))
train_ds = DecoderDataset(datasets['train'])
test_ds = DecoderDataset(datasets['test'])

print('Dataset lengths: ', len(train_ds), len(test_ds))

# Create attention mask from lengths
def create_attention_mask_from_lengths(lengths, max_len=None):
    if max_len is None:
        max_len = max(lengths)
    batch_size = len(lengths)
    mask = torch.zeros(batch_size, max_len)
    for i, length in enumerate(lengths):
        mask[i, :length] = 1
    return mask.bool()

def collate_2(batch):
    X, transcriptions = zip(*batch)

    X_padded = pad_sequence(X, batch_first=True, padding_value=0)
    X_padded = torch.cat([X_padded, torch.zeros(X_padded.size(0), 1024-X_padded.size(1), X_padded.size(2))], dim=1)

    X_padded = X_padded.reshape(-1, 256, 1024)
    X_mask = X_padded.ne(0).all(-1)

    transcription_ids = [torch.tensor([ph2id[p] for p in sent]) for sent in transcriptions]
    transcription_lens = torch.tensor([len(t) for t in transcription_ids])

    transcription_ids_y = pad_sequence(transcription_ids, batch_first=True, padding_value=-100)
    transcription_ids_x = pad_sequence(transcription_ids, batch_first=True, padding_value=ph2id['<PAD>'])

    transcription_mask = create_attention_mask_from_lengths(transcription_lens, max_len=transcription_ids_y.shape[1])

    return X_padded, X_mask, transcription_ids_x, transcription_ids_y, transcription_lens, transcription_mask, transcriptions

# DataLoader
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=32, collate_fn=collate_2)
eval_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=32, collate_fn=collate_2)

# Move model to device
model.to(device)

# Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=5e-4)
# scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=1000)
# scheduler = create_lr_scheduler_with_warmup(scheduler,
#                                             warmup_start_value=0.001,
#                                             warmup_end_value=0.1,
#                                             warmup_duration=15)
loss_fn = nn.CrossEntropyLoss()
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=250,
    num_training_steps=len(train_loader)*epochs,
)

# Training and evaluation loop
for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    train_cer = 0
    total_edit_distance = 0
    total_seq_length = 0
    progress = tqdm(range(len(train_loader)))

    for batch_idx, (neuro, neuro_mask, text_ids_x, text_ids, text_lens, text_mask, gt_texts) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Train {epoch}'):
        neuro = neuro.to(device)
        neuro_mask = neuro_mask.to(device)
        text_ids_x = text_ids_x.to(device)
        text_ids = text_ids.to(device)
        text_lens = text_lens.to(device)
        text_mask = text_mask.to(device)
        optimizer.zero_grad()

        # Add noise to input
        # neuro += torch.randn(neuro.shape, device=device) * 0.8
        # neuro += (torch.randn([neuro.shape[0], 1, neuro.shape[2]], device=device) * 0.2)

        # Forward pass
        outputs = model(neuro, neuro_mask, text_ids_x, text_ids, text_mask)

        predicted_token_ids = outputs.logits.argmax(-1)
        seqs = [[id2ph[ph.item()] for ph in predicted_token_ids[i]] for i in range(len(predicted_token_ids))]
        gt_seqs = [[id2ph[ph.item()] for ph in text_ids_x[i]] for i in range(len(text_ids_x))]

        gt_seqs = [s[1:text_lens[i]] for i, s in enumerate(gt_seqs)]
        seqs = [s[:text_lens[i]-1] for i, s in enumerate(seqs)]

        if total_train_loss == 0:
            for i in range(3):
                print('PR: ', seqs[i])
                print('GT: ', gt_seqs[i])
        for i in range(len(seqs)):
            total_edit_distance += SequenceMatcher(a=seqs[i], b=gt_seqs[i]).distance()
            total_seq_length += len(gt_seqs[i])

        # Compute loss
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        total_train_loss += loss.item()
        progress.update()
        progress.set_description(
            f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {total_train_loss/(batch_idx+1)} | CER: {total_edit_distance/total_seq_length}"
        )


    print()
    avg_train_loss = total_train_loss / len(train_loader)
    train_cer = total_edit_distance / total_seq_length

    print(f"Epoch {epoch+1}, Average Training Loss: {avg_train_loss} | CER: {train_cer}")
    if USE_WANDB:
        wandb.log({"train/loss": avg_train_loss, "train/cer_tf": train_cer, 'train/lr': optimizer.param_groups[0]['lr']})

    model.eval()
    total_eval_loss = 0
    eval_cer = 0
    total_edit_distance = 0
    total_seq_length = 0

    with torch.no_grad():
        for neuro, neuro_mask, text_ids_x, text_ids, text_lens, text_mask, gt_texts in tqdm(eval_loader, desc=f'Eval {epoch}'):
            neuro = neuro.to(device)
            neuro_mask = neuro_mask.to(device)
            text_ids_x = text_ids_x.to(device)
            text_ids = text_ids.to(device)
            text_lens = text_lens.to(device)
            text_mask = text_mask.to(device)
            outputs = model(neuro, neuro_mask, text_ids_x, text_ids, text_mask)
            loss = outputs.loss
            total_eval_loss += loss.item()

            predicted_token_ids = outputs.logits.argmax(-1)

            seqs = [[id2ph[ph.item()] for ph in predicted_token_ids[i]] for i in range(len(predicted_token_ids))]
            gt_seqs = [[id2ph[ph.item()] for ph in text_ids_x[i]] for i in range(len(text_ids_x))]

            gt_seqs = [s[1:text_lens[i]] for i, s in enumerate(gt_seqs)]
            seqs = [s[:text_lens[i]-1] for i, s in enumerate(seqs)]

            if total_train_loss == 0:
                for i in range(3):
                    print('PR: ', seqs[i])
                    print('GT: ', gt_seqs[i])
            for i in range(len(seqs)):
                total_edit_distance += SequenceMatcher(a=seqs[i], b=gt_seqs[i]).distance()
                total_seq_length += len(gt_seqs[i])

    avg_eval_loss = total_eval_loss / len(eval_loader)
    eval_cer = total_edit_distance / total_seq_length

    print(f"Epoch {epoch+1}, Average Evaluation Loss: {avg_eval_loss} | CER: {eval_cer}")
    if USE_WANDB:
        wandb.log({"eval/loss": avg_eval_loss, "eval/cer_tf": eval_cer})

    torch.save({
        'model_state_dict': model.state_dict(),
    }, f'{save_dir}/{epoch}.pth')
