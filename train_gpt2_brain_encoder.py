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
from transformers import GPT2LMHeadModel, GPT2Config
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from itertools import groupby
from models import GRUEncoder
import string

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

# Initialize the GRU Encoder
encoder = GRUEncoder(output_embed_dim=768, layer_dim=3) # len(tokens)

# Load the GPT-2 model with cross-attention enabled
config = GPT2Config.from_pretrained('gpt2')
config.vocab_size = len(tokens)
config.add_cross_attention = True
config.pad_token_id = ph2id['<PAD>']
config.eos_token_id = ph2id['<EOS>']

decoder = GPT2LMHeadModel.from_pretrained('./phoneme_lm_gpt2_pretr/checkpoint-330000/', config=config)

for name, param in decoder.named_parameters():
    if 'crossattention' not in name or 'cross_attn' not in name:
        param.requires_grad = False

print(encoder)
print(decoder)

# Experiment settings
exp_name = 'gru_gpt2_attention_pretr_frozen'
wandb.init(project='huggingface', name=exp_name)

device = 'cuda'
epochs = 1000
batch_size = 96

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

# Collate function
def collate(batch):
    X, transcriptions = zip(*batch)
    X_lens = [x.shape[0] for x in X]

    X_padded = pad_sequence(X, batch_first=True, padding_value=0).to(device)
    X_mask = create_attention_mask_from_lengths(X_lens, max_len=X_padded.shape[1]).to(device)

    transcription_ids = [torch.tensor([ph2id[p] for p in sent]) for sent in transcriptions]
    transcription_lens = torch.tensor([len(t) for t in transcription_ids], device=device).to(device)

    transcription_ids_y = pad_sequence(transcription_ids, batch_first=True, padding_value=-100).to(device)
    transcription_ids_x = pad_sequence(transcription_ids, batch_first=True, padding_value=ph2id['<PAD>']).to(device)

    transcription_mask = create_attention_mask_from_lengths(transcription_lens, max_len=transcription_ids_y.shape[1]).to(device)
    
    return X_padded.to(device), X_mask.to(device), transcription_ids_x.to(device), transcription_ids_y.to(device), transcription_lens.to(device), transcription_mask.to(device), transcriptions

# DataLoader
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate)
eval_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate)

# Move models to device
encoder.to(device)
decoder.to(device)

# Optimizer and scheduler
optimizer = AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-5)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=10000)
loss_fn = nn.CrossEntropyLoss()

# Compute WER
def compute_wer(predictions, references):
    normalizer = BasicTextNormalizer()
    normalized_predictions = [normalizer(pred) for pred in predictions]
    normalized_references = [normalizer(ref) for ref in references]
    wer = jiwer.wer(normalized_references, normalized_predictions)
    return 100 * wer

# Training and evaluation loop
for epoch in range(epochs):
    encoder.train()
    decoder.train()
    total_train_loss = 0
    train_wer = 0
    train_cer = 0
    total_edit_distance = 0
    total_seq_length = 0

    for neuro, neuro_mask, text_ids_x, text_ids, text_lens, text_mask, gt_texts in tqdm(train_loader, desc=f'Train {epoch}'):
        optimizer.zero_grad()

        # Add noise to input
        neuro += torch.randn(neuro.shape, device=device) * 0.8
        neuro += (torch.randn([neuro.shape[0], 1, neuro.shape[2]], device=device) * 0.2)

        # Encoder forward pass
        encoder_outputs = encoder(neuro)
        encoder_outputs_mask = create_attention_mask_from_lengths(((neuro_mask.sum(dim=1) - encoder.kernelLen) / encoder.strideLen).to(torch.int32), max_len=encoder_outputs.shape[1]).to(device)

        # Decoder forward pass with cross-attention
        decoder_outputs = decoder(input_ids=text_ids_x, attention_mask=text_mask, encoder_hidden_states=encoder_outputs, encoder_attention_mask=encoder_outputs_mask, labels=text_ids)

        if total_train_loss == 0:
            predicted_token_ids = decoder_outputs.logits.argmax(-1)
            seqs = [[id2ph[ph.item()] for ph in predicted_token_ids[i] if id2ph[ph.item()] not in ('<PAD>', '<EOS>')] for i in range(len(predicted_token_ids))]
            gt_seqs = [[id2ph[ph.item()] for ph in text_ids_x[i] if id2ph[ph.item()] not in ('<PAD>', '<EOS>')] for i in range(len(text_ids_x))]
            for i in range(3):
                print(seqs[i])
                print(gt_seqs[i])
            

        # Compute loss
        loss = decoder_outputs.loss
        
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        
        # seqs = [[id2ph[ph.item()] for ph in predicted_token_ids[i] if id2ph[ph.item()] not in ('<PAD>', '<EOS>')] for i in range(len(predicted_token_ids))]
        # gt_seqs = [[id2ph[ph.item()] for ph in text_ids_x[i] if id2ph[ph.item()] not in ('<PAD>', '<EOS>')] for i in range(len(text_ids_x))]
        
        # train_wer += compute_wer(seqs, gt_seqs)

        # for gt, pred in zip(gt_seqs, seqs):
        #     matcher = SequenceMatcher(a=gt, b=pred)
        #     total_edit_distance += matcher.distance()
        #     total_seq_length += len(gt)

    train_cer = 0#  total_edit_distance / total_seq_length
    avg_train_loss = total_train_loss / len(train_loader)
    train_wer = train_wer / len(train_loader)
    
    print(f"Epoch {epoch+1}, Average Training Loss: {avg_train_loss}, WER: {train_wer}, CER: {train_cer}")
    wandb.log({"train/loss": avg_train_loss, 'train/wer': train_wer, 'train/cer': train_cer})

    encoder.eval()
    decoder.eval()
    total_eval_loss = 0
    eval_wer = 0
    eval_cer = 0
    total_edit_distance = 0
    total_seq_length = 0

    with torch.no_grad():
        for neuro, neuro_mask, text_ids_x, text_ids, text_lens, text_mask, gt_texts in tqdm(eval_loader, desc=f'Eval {epoch}'):
            encoder_outputs = encoder(neuro)
            encoder_outputs_mask = create_attention_mask_from_lengths(((neuro_mask.sum(dim=1) - encoder.kernelLen) / encoder.strideLen).to(torch.int32), max_len=encoder_outputs.shape[1]).to(device)
            decoder_outputs = decoder(input_ids=text_ids_x, attention_mask=text_mask, encoder_hidden_states=encoder_outputs, encoder_attention_mask=encoder_outputs_mask, labels=text_ids)
            
            loss = decoder_outputs.loss
            total_eval_loss += loss.item()

            decoder_outputs = decoder_outputs.logits

            # predicted_token_ids = decoder_outputs.argmax(-1)
            # seqs = [[id2ph[ph.item()] for ph in predicted_token_ids[i] if id2ph[ph.item()] != '_'] for i in range(len(predicted_token_ids))]
            # pred_texts = [''.join(remove_consecutive_duplicates(seq)) for seq in seqs]
            # print(f'GT: {gt_texts[0]}\nPRED: {pred_texts[0]}')

            # eval_wer += compute_wer(pred_texts, gt_texts)

            # for gt, pred in zip(gt_texts, pred_texts):
            #     matcher = SequenceMatcher(a=gt, b=pred)
            #     total_edit_distance += matcher.distance()
            #     total_seq_length += len(gt)

    eval_cer = 0 # total_edit_distance / total_seq_length
    avg_eval_loss = total_eval_loss / len(eval_loader)
    eval_wer = eval_wer / len(eval_loader)
    
    print(f"Epoch {epoch+1}, Average Evaluation Loss: {avg_eval_loss}, WER: {eval_wer}, CER: {eval_cer}")
    wandb.log({"eval/loss": avg_eval_loss, "eval/wer": eval_wer, 'eval/cer': eval_cer})

    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
    }, f'{save_dir}/{epoch}.pth')
