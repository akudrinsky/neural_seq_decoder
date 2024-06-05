from datasets import load_dataset
import re
from speechbrain.inference.text import GraphemeToPhoneme
from num2words import num2words
import unicodedata
from multiprocess import Pool, set_start_method
import torch
import os

# Define preprocessing functions
def remove_words_with_numbers(text):
    pattern = r'\w*\d\w*'
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text

def convert_numbers_to_words(text):
    def replace_with_words(match):
        number = match.group(0)
        try:
            return num2words(number, lang='en')
        except:
            return number
    return re.sub(r'\b\d+\b', replace_with_words, text)

def remove_accents(input_text):
    nfkd_form = unicodedata.normalize('NFKD', input_text)
    return ''.join([char for char in nfkd_form if not unicodedata.combining(char)])

def preprocess_text(example):
    text = example['text']
    text = text.replace('<unk>', '')
    text = remove_words_with_numbers(text)
    text = convert_numbers_to_words(text)
    text = remove_accents(text)
    text = re.sub(r'[^\x00-\x7F]+|[^\w\s.]|\n|\t', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    sentences = [sentence for sentence in re.split(r'(?<=[.!?]) +', text) if 10 <= len(sentence.split()) <= 100]
    sentences = [s.replace('.', '').strip() for s in sentences]
    return {'sentences': sentences}

# Load dataset
splits = ['train', 'validation']
datasets = load_dataset("wikitext", "wikitext-2-v1", split=splits)
datasets = {split: ds for split, ds in zip(splits, datasets)}

# Preprocess datasets
processed_datasets = {split: ds.map(preprocess_text, batched=False, remove_columns=['text']) for split, ds in datasets.items()}

for split in processed_datasets:
    processed_datasets[split] = processed_datasets[split].map(lambda x: {'sentences': sum(x['sentences'], [])}, batched=True, batch_size=1000, num_proc=32)

def create_subset(dataset, subset_ratio=0.1, seed=42):
    shuffled_dataset = dataset.shuffle(seed=seed)

    num_samples = int(len(shuffled_dataset) * subset_ratio)

    subset = shuffled_dataset.select(range(num_samples))

    return subset

for split in splits:
    processed_datasets[split] = create_subset(processed_datasets[split], subset_ratio=0.05)

# Function to create subsets
def create_subsets(dataset, num_parts):
    shuffled_dataset = dataset.shuffle(seed=42)
    part_size = len(shuffled_dataset) // num_parts
    subsets = [shuffled_dataset.select(range(i * part_size, (i + 1) * part_size)) for i in range(num_parts)]
    return subsets

# Split datasets into 4 equal parts
split_subsets = {split: create_subsets(ds, 4) for split, ds in processed_datasets.items()}

# Function to process each subset independently
def process_subset(subset, subset_index, split):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(subset_index % torch.cuda.device_count())
    print(f"Processing {split} subset {subset_index} on GPU: {os.environ['CUDA_VISIBLE_DEVICES']} with PID: {os.getpid()}")

    # Load the G2P model within the subprocess to ensure correct CUDA device assignment
    g2p = GraphemeToPhoneme.from_hparams("speechbrain/soundchoice-g2p", savedir="pretrained_models/soundchoice-g2p", run_opts={"device":"cuda"})

    def phoneme_extraction(example):
        phonemes = g2p(example['sentences'])
        return {'phonemes': phonemes}

    subset = subset.map(phoneme_extraction, batched=True, batch_size=64, num_proc=1)
    subset.save_to_disk(f"./wikitext_full-phonemes-{split}-part{subset_index}")
    return f"Finished processing {split} subset {subset_index}"

# Parallel processing
def process_all_subsets():
    set_start_method("spawn")
    pool = Pool(processes=8)
    results = []

    for split in splits:
        for i, subset in enumerate(split_subsets[split]):
            result = pool.apply_async(process_subset, (subset, i, split))
            results.append(result)

    for result in results:
        print(result.get())

    pool.close()
    pool.join()

if __name__ == "__main__":
    process_all_subsets()
