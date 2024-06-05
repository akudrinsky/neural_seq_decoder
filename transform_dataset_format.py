import torch
from tqdm import tqdm
import pickle

with open("/mnt/scratch/kudrinsk/eval_challenge/ptDecoder_ctc", "rb") as handle:
    loadedData = pickle.load(handle)

new_dataset = []
for split_name in loadedData:
    for day in tqdm(range(len(loadedData[split_name]))):
        for trial in range(len(loadedData[split_name][day]["sentenceDat"])):
            sample = {
                "neuro": loadedData[split_name][day]["sentenceDat"][trial],
                "neuro_len": loadedData[split_name][day]["sentenceDat"][trial].shape[0],
                "text": loadedData[split_name][day]["transcriptions"][trial],
                "day": day,
                "split_name": split_name,
                "phonemes": loadedData[split_name][day]["phonemes"][trial],
            }
            new_dataset.append(sample)

with open("/mnt/scratch/kudrinsk/eval_challenge/dataset_phonemes.pickle", "wb") as handle:
    pickle.dump(new_dataset, handle)