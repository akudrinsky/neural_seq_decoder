"""
Preprocess dataset downloaded from https://datadryad.org/stash/dataset/doi:10.5061/dryad.x69p8czpq
"""


from tqdm import tqdm
import pickle
import argparse
from .original_preprocessing_code import original_prep_code

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, required=True, default="/mnt/scratch/kudrinsk/eval_challenge/competitionData/")
parser.add_argument("--output_path", type=str, required=True, default="/mnt/scratch/kudrinsk/eval_challenge/dataset_processed.pickle")

args = parser.parse_args()

prepr_data = original_prep_code(args.input_path)

new_dataset = []
for split_name in prepr_data:
    for day in tqdm(range(len(prepr_data[split_name])), desc=f"Processing {split_name}"):
        for trial in range(len(prepr_data[split_name][day]["sentenceDat"])):
            sample = {
                "neuro": prepr_data[split_name][day]["sentenceDat"][trial],
                "neuro_len": prepr_data[split_name][day]["sentenceDat"][trial].shape[0],
                "text": prepr_data[split_name][day]["transcriptions"][trial],
                "day": day,
                "split_name": split_name,
            }
            new_dataset.append(sample)

with open(args.output_path, "wb") as handle:
    pickle.dump(new_dataset, handle)
