from tqdm import tqdm
import pickle
import argparse
from speechbrain.inference.text import GraphemeToPhoneme

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, required=True, default="/mnt/scratch/kudrinsk/eval_challenge/dataset_processed.pickle")
parser.add_argument("--output_path", type=str, required=True, default="/mnt/scratch/kudrinsk/eval_challenge/dataset_phonemes.pickle")

args = parser.parse_args()

with open(args.input_path, "rb") as handle:
    data = pickle.load(handle)

g2p = GraphemeToPhoneme.from_hparams("speechbrain/soundchoice-g2p", savedir="pretrained_models/soundchoice-g2p")

for sample in tqdm(data, desc="Processing samples"):
    words = sample['text']
    phonemes = g2p(words)

    sample['phonemes'] = phonemes

with open(args.output_path, "wb") as handle:
    pickle.dump(data, handle)
