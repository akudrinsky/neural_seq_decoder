from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
import torch
import string
import os
import pickle

import hydra
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from neural_decoder.model import GRUDecoder

def loadModel(modelDir, nInputLayers=24, device="cuda"):
    modelWeightPath = modelDir + "/modelWeights"
    with open(modelDir + "/args", "rb") as handle:
        args = pickle.load(handle)

    model = GRUDecoder(
        neural_dim=args["nInputFeatures"],
        n_classes=args["nClasses"],
        hidden_dim=args["nUnits"],
        layer_dim=args["nLayers"],
        nDays=nInputLayers,
        dropout=args["dropout"],
        device=device,
        strideLen=args["strideLen"],
        kernelLen=args["kernelLen"],
        gaussianSmoothWidth=args["gaussianSmoothWidth"],
        bidirectional=args["bidirectional"],
    ).to(device)

    model.load_state_dict(torch.load(modelWeightPath, map_location=device))
    return model

id2ph_ = [
    'AA', 'AE', 'AH', 'AO', 'AW',
    'AY', 'B',  'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V',
    'W', 'Y', 'Z', 'ZH', ' '
]

def id2ph(i):
    return id2ph_[i-1]

len(id2ph_)

class InferPipe():
    def __init__(self, 
                 b2p_path = '/mnt/scratch/kudrinsk/eval_challenge/trains/oldcode_characters_lrfactor200_3layers/', 
                 p2g_path = './p2g/checkpoint-34000/',
                 device = 'cuda',
                ):
        tokens = ['<PAD>', '<EOS>', '<START_DEC>', '<UNK>', ' '] + list(string.ascii_lowercase) + self.get_p2g_phoneme_list()
        self.token_to_index = {token: idx for idx, token in enumerate(tokens)}
        self.index_to_token = {idx: token for token, idx in self.token_to_index.items()}
        self.device = device
        
        # Load the trained model
        config = T5Config(
            vocab_size=len(tokens),
            num_layers=6,
            pad_token_id=self.token_to_index['<PAD>'], 
            eos_token_id=self.token_to_index['<EOS>'],
            decoder_start_token_id=self.token_to_index['<START_DEC>'],
        )
        self.p2g = T5ForConditionalGeneration.from_pretrained(p2g_path)
        self.p2g.eval().to(self.device)  # Set the model to evaluation mode

        self.b2p = loadModel(b2p_path)
        self.b2p.eval().to(self.device)

    def get_p2g_phoneme_list(self):
        phonemes_list = [
            'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 
            'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 
            'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH', 'spn'
        ]
        return phonemes_list

    def phonemes_to_sentence(self, phonemes):
        # Convert phonemes to input IDs
        input_ids = [self.token_to_index[ph] for ph in phonemes if ph in self.token_to_index]
        input_ids = torch.tensor(input_ids).unsqueeze(0).to(self.device) #.cuda()  # Add batch dimension and move to GPU
    
        # Generate sentence
        with torch.no_grad():
            outputs = self.p2g.generate(
                input_ids=input_ids,
                max_length=100,
                num_beams=10,
                # early_stopping=True,
                # length_penalty=10.0,  # You can adjust this to penalize longer sequences
                # no_repeat_ngram_size=2  # Ensure no repeated 2-grams
            )
        generated_ids = outputs[0].cpu().tolist()
        if self.token_to_index['<EOS>'] in generated_ids:
            eos_index = generated_ids.index(self.token_to_index['<EOS>'])
            generated_ids = generated_ids[1:eos_index] # first is <BOS>
    
        sentence = ''.join([self.index_to_token[idx] for idx in generated_ids if idx in self.index_to_token])
    
        return sentence
        
    def infer_sample(self, brain_data, day_idx):
        with torch.no_grad():
            out = self.b2p.forward(brain_data.unsqueeze(0).to(self.device), day_idx.unsqueeze(0).to(self.device)).cpu()

        adjusted_neuro_len = int((brain_data.shape[0] - self.b2p.kernelLen) / self.b2p.strideLen)
        
        decodedSeq = torch.argmax(
            torch.tensor(out[0:adjusted_neuro_len, :]),
            dim=-1,
        )
        decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
        decodedSeq = decodedSeq.cpu().detach().numpy()[0]
        decodedSeq = np.array([i for i in decodedSeq if i != 0])
        decoded_phonemes = [id2ph(p) for p in decodedSeq]

        decoded_phonemes.append('<EOS>')

        sentence = self.phonemes_to_sentence(decoded_phonemes)

        return sentence

pipe = InferPipe()

from neural_decoder.dataset import SpeechDataset
import pickle
from tqdm import tqdm

split_name = 'test' # train, competition, test

with open('/mnt/scratch/kudrinsk/eval_challenge/ptDecoder_ctc', "rb") as handle:
    loadedData = pickle.load(handle)
print(loadedData.keys())

dataset = SpeechDataset(loadedData[split_name])

texts = []
gt_texts = []

for i in tqdm(range(min(len(dataset), 5))):
    neuro, phonemes, neurolen, phlen, day_idx, trans = dataset[i]

    gt_texts.append(trans)
    texts.append(pipe.infer_sample(neuro, day_idx))

with open(f'results_{split_name}_pred.txt', 'w') as file:
    for text in texts:
        file.write(text + '\n')

with open(f'results_{split_name}_gt.txt', 'w') as file:
    for text in gt_texts:
        file.write(text + '\n')