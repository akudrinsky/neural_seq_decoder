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
from transformers import GPT2LMHeadModel, GPT2Config, get_linear_schedule_with_warmup, BertModel, BertConfig

USE_WANDB = True

# Define the phoneme list and mappings
def get_phoneme_list():
    phonemes_list = [
        'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY',
        'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P',
        'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH', 'spn'
    ]
    return phonemes_list

tokens = ['<PAD>', '<EOS>', '<BOS>', '<UNK>', ' '] + get_phoneme_list()
print(f'Number of tokens: {len(tokens)}')
id2ph = {i: ph for i, ph in enumerate(tokens)}
ph2id = {ph: i for i, ph in enumerate(tokens)}

print(tokens)

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
            max_position_embeddings=1024,
        )
        self.bert = BertModel(config)

    def forward(self, x):
        x = self.linear(x)
        return self.bert(inputs_embeds=x).last_hidden_state

# Define the model class with concatenated embeddings
class Seq2SeqModel(nn.Module):
    def __init__(self, encoder_type='linear'):
        super(Seq2SeqModel, self).__init__()

        config = GPT2Config.from_pretrained('gpt2')
        config.vocab_size = len(tokens)
        config.add_cross_attention = True
        config.pad_token_id = ph2id['<PAD>']
        config.eos_token_id = ph2id['<EOS>']
        config.bos_token_id = ph2id['<BOS>']

        self.decoder = GPT2LMHeadModel.from_pretrained('./phoneme_lm_gpt2_boseos/checkpoint-200000/', config=config)
        self.decoder.config.pad_token_id = ph2id['<PAD>']
        self.decoder.config.eos_token_id = ph2id['<EOS>']
        self.decoder.config.bos_token_id = ph2id['<BOS>']

        self.encoder_type = encoder_type
        if encoder_type == 'linear':
            self.neuro_lm = nn.Linear(256, 768)
        else:
            self.neuro_lm = BERTEncoder(input_dim=256, hidden_dim=768, layer_dim=2) # nn.Linear(256, config.n_embd)

    def forward(self, neuro, neuro_mask, text_ids_x, text_ids, text_mask):
        batch_size, neuro_time = neuro.shape[0], neuro.shape[1]

        neuro = self.neuro_lm(neuro)

        # Concatenate neuro features with token embeddings
        token_embeddings = self.decoder.transformer.wte(text_ids_x)
        print(neuro.shape, token_embeddings.shape)
        combined_embeddings = torch.cat((neuro, token_embeddings), dim=1)

        # Adjust text_mask shape
        text_mask = torch.cat((neuro_mask, text_mask), dim=1)

        # Adjust labels' shape
        extra_labels = torch.full((batch_size, neuro_time), -100, device=text_ids.device)
        text_ids = torch.cat((extra_labels, text_ids), dim=1)

        # Forward pass through decoder
        # print(combined_embeddings.shape, text_mask.shape, text_ids.shape)
        decoder_outputs = self.decoder(inputs_embeds=combined_embeddings, attention_mask=text_mask, labels=text_ids)

        loss, logits = decoder_outputs.loss, decoder_outputs.logits
        logits = logits[:, neuro_time:, :]

        return logits, loss

    def generate(self, neuro, attention_mask=None, max_length=None, **kwargs):
        batch_size = neuro.size(0)
        neuro_time = neuro.size(1)

        # Project neuro features
        neuro = self.neuro_lm(neuro)

        # Initial token embeddings
        bos_token_id = self.decoder.config.bos_token_id
        bos_token_tensor = torch.full((batch_size, 1), bos_token_id, device=neuro.device)

        # Concatenate neuro features with initial token embeddings
        token_embeddings = self.decoder.transformer.wte(bos_token_tensor)
        combined_embeddings = torch.cat((neuro, token_embeddings), dim=1)

        # print(neuro.shape, combined_embeddings.shape, attention_mask.shape)

        # Adjust attention mask shape
        if attention_mask is not None:
            extra_mask = torch.ones((batch_size, 1), device=attention_mask.device) # add mask for BOS token
            attention_mask = torch.cat((extra_mask, attention_mask), dim=1)

        # print(combined_embeddings.shape, attention_mask.shape)

        generated_ids = self.decoder.generate(
            inputs_embeds=combined_embeddings,
            attention_mask=attention_mask,
            pad_token_id=self.decoder.config.pad_token_id,
            eos_token_id=self.decoder.config.eos_token_id,
            bos_token_id=self.decoder.config.bos_token_id,
            max_length=max_length + neuro_time,
            **kwargs)

        # print(generated_ids.shape)

        # generated_ids = generated_ids[:, neuro_time:]
        return generated_ids

# Create the Seq2Seq model
model = Seq2SeqModel()
# model.load_state_dict(torch.load('../ckpts/concat_model/3.pth')['model_state_dict'])

# Experiment settings
exp_name = 'concat_model_left_bert'

if USE_WANDB:
    wandb.init(project='ctc', name=exp_name)
    wandb.save(os.path.abspath(__file__))

device = 'cuda'
epochs = 1000
batch_size = 12

save_dir = f'../ckpts/{exp_name}/'
os.makedirs(save_dir, exist_ok=True)

# Load the dataset
with open("/mnt/scratch/kudrinsk/eval_challenge/dataset_phonemes.pickle", "rb") as handle:
    data = pickle.load(handle)

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

def pad_sequence_left(sequences, batch_first=False, padding_value=0):
    # Reverse each sequence
    reversed_sequences = [torch.flip(seq, dims=[0]) for seq in sequences]

    # Pad the reversed sequences
    padded_reversed_sequences = pad_sequence(reversed_sequences, batch_first=batch_first, padding_value=padding_value)

    # Reverse the padded sequences back to the original order
    if batch_first:
        padded_sequences = torch.flip(padded_reversed_sequences, dims=[1])
    else:
        padded_sequences = torch.flip(padded_reversed_sequences, dims=[0])

    return padded_sequences

def collate_2(batch):
    X, transcriptions = zip(*batch)

    X_padded = pad_sequence_left(X, batch_first=True, padding_value=0)
    maxlen = 925 # pretrained gpt2 has context of 1024, and we concat phonemes
    if X_padded.size(1) < maxlen:
        X_padded = torch.cat([torch.zeros(X_padded.size(0), maxlen-X_padded.size(1), X_padded.size(2)), X_padded], dim=1)
    elif X_padded.shape[1] > maxlen:
        X_padded = X_padded[-maxlen:]

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
    total_edit_distance_gen = 0
    total_seq_length = 0
    total_seq_length_gen = 0.00001
    progress = tqdm(range(len(train_loader)))

    for batch_idx, (neuro, neuro_mask, text_ids_x, text_ids, text_lens, text_mask, gt_texts) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Train {epoch}'):
        # if total_train_loss != 0:
            # continue
        neuro = neuro.to(device)
        neuro_mask = neuro_mask.to(device)
        text_ids_x = text_ids_x.to(device)
        text_ids = text_ids.to(device)
        text_lens = text_lens.to(device)
        text_mask = text_mask.to(device)
        optimizer.zero_grad()

        # Forward pass
        logits, loss = model(neuro, neuro_mask, text_ids_x, text_ids, text_mask)

        predicted_token_ids = logits.argmax(-1)
        seqs = [[id2ph[ph.item()] for ph in predicted_token_ids[i]] for i in range(len(predicted_token_ids))]
        gt_seqs = [[id2ph[ph.item()] for ph in text_ids_x[i]] for i in range(len(text_ids_x))]

        gt_seqs = [s[1:text_lens[i]] for i, s in enumerate(gt_seqs)]
        seqs = [s[:text_lens[i]-1] for i, s in enumerate(seqs)]

        for i in range(len(seqs)):
            total_edit_distance += SequenceMatcher(a=seqs[i], b=gt_seqs[i]).distance()
            total_seq_length += len(gt_seqs[i])

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        total_train_loss += loss.item()

        # generate
        # generated_ids = model.generate(neuro, attention_mask=neuro_mask, max_length=100)
        # gen_seqs = [[id2ph[ph.item()] for ph in gen_seq] for gen_seq in generated_ids]

        # for i in range(len(gen_seqs)):
        #     total_edit_distance_gen += SequenceMatcher(a=gen_seqs[i], b=gt_seqs[i]).distance()
        #     total_seq_length_gen += len(gt_seqs[i])

        progress.update()
        progress.set_description(
            f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {total_train_loss/(batch_idx+1)} | CER_tf: {total_edit_distance/total_seq_length} | CER: {total_edit_distance_gen/total_seq_length_gen}"
        )


    print()
    avg_train_loss = total_train_loss / len(train_loader)
    train_cer = total_edit_distance / total_seq_length
    train_cer_gen = total_edit_distance_gen / total_seq_length_gen

    print(f"Epoch {epoch+1}, Average Training Loss: {avg_train_loss} | CER (TF): {train_cer} | CER (Gen): {train_cer_gen}")
    if USE_WANDB:
        wandb.log({"train/loss": avg_train_loss, "train/cer_tf": train_cer, "train/cer": train_cer_gen, 'train/lr': optimizer.param_groups[0]['lr']})

    model.eval()
    total_eval_loss = 0
    eval_cer = 0
    total_edit_distance = 0
    total_edit_distance_gen = 0
    total_seq_length = 0.000001
    total_seq_length_gen = 0.00001

    with torch.no_grad():
        for neuro, neuro_mask, text_ids_x, text_ids, text_lens, text_mask, gt_texts in tqdm(eval_loader, desc=f'Eval {epoch}'):
            neuro = neuro.to(device)
            neuro_mask = neuro_mask.to(device)
            text_ids_x = text_ids_x.to(device)
            text_ids = text_ids.to(device)
            text_lens = text_lens.to(device)
            text_mask = text_mask.to(device)
            logits, loss = model(neuro, neuro_mask, text_ids_x, text_ids, text_mask)
            total_eval_loss += loss.item()

            predicted_token_ids = logits.argmax(-1)

            seqs = [[id2ph[ph.item()] for ph in predicted_token_ids[i]] for i in range(len(predicted_token_ids))]
            gt_seqs = [[id2ph[ph.item()] for ph in text_ids_x[i]] for i in range(len(text_ids_x))]

            gt_seqs = [s[1:text_lens[i]] for i, s in enumerate(gt_seqs)]
            seqs = [s[:text_lens[i]-1] for i, s in enumerate(seqs)]

            for i in range(len(seqs)):
                total_edit_distance += SequenceMatcher(a=seqs[i], b=gt_seqs[i]).distance()
                total_seq_length += len(gt_seqs[i])

            # generate
            generated_ids = model.generate(neuro, attention_mask=neuro_mask, max_length=100)
            gen_seqs = [[id2ph[ph.item()] for ph in gen_seq if id2ph[ph.item()] not in ('<BOS>', '<PAD>')] for gen_seq in generated_ids]

            for i in range(len(gen_seqs)):
                total_edit_distance_gen += SequenceMatcher(a=gen_seqs[i], b=gt_seqs[i]).distance()
                total_seq_length_gen += len(gt_seqs[i])

            for i in range(1):
                print('GT: ', gt_seqs[i])
                print('PR: ', seqs[i])
                print('GEN: ', gen_seqs[i])

    avg_eval_loss = total_eval_loss / len(eval_loader)
    eval_cer = total_edit_distance / total_seq_length
    eval_cer_gen = total_edit_distance_gen / total_seq_length_gen

    print(f"Epoch {epoch+1}, Average Evaluation Loss: {avg_eval_loss} | CER (TF): {eval_cer} | CER (Gen): {eval_cer_gen}")
    if USE_WANDB:
        wandb.log({"eval/loss": avg_eval_loss, "eval/cer_tf": eval_cer, "eval/cer": eval_cer_gen})

    torch.save({
        'model_state_dict': model.state_dict(),
    }, f'{save_dir}/{epoch}.pth')
