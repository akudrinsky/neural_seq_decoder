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
import torchaudio

from torchaudio.models.decoder import ctc_decoder
from torchaudio.models.decoder import download_pretrained_files

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

phoneme_list = ['<blank>', ' ', 'e', 't', 'a', 'o', 'n', 'i', 'h', 's', 'r', 'd', 'l', 'u', 'm', 'w', 'c', 'f', 'g', 'y', 'p', 'b', 'v', 'k', "'", 'x', 'j', 'q', 'z']
id2ph = {i: ph for i, ph in enumerate(phoneme_list)}
ph2id = {ph: i for i, ph in enumerate(phoneme_list)}

class InferPipe():
    def __init__(self, 
                 b2p_path = '/mnt/scratch/kudrinsk/eval_challenge/trains/oldcode_letters/', 
                 device = 'cuda',
                ):
        self.device = device
        
        LM_WEIGHT = 3.23
        WORD_SCORE = -0.26
        self.files = download_pretrained_files("librispeech-4-gram") # librispeech-4-gram
        self.beam_search_decoder = ctc_decoder(
            lexicon=self.files.lexicon,
            tokens=self.files.tokens,
            lm=self.files.lm,
            nbest=3,
            beam_size=1500,
            lm_weight=LM_WEIGHT,
            word_score=WORD_SCORE,
        )

        self.b2p = loadModel(b2p_path)
        self.b2p.eval().to(self.device)

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

        adjusted_out = out[0:adjusted_neuro_len, :]
        
        beam_search_result = self.beam_search_decoder(adjusted_out)
        beam_search_transcript = " ".join(beam_search_result[0][0].words).strip()
    
        return beam_search_transcript

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