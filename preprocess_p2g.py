from datasets import load_dataset
import re
from speechbrain.inference.text import GraphemeToPhoneme
from num2words import num2words
import unicodedata

g2p = GraphemeToPhoneme.from_hparams("speechbrain/soundchoice-g2p", savedir="pretrained_models/soundchoice-g2p")

def remove_words_with_numbers(text):
    # Regex pattern to find words containing digits
    pattern = r'\w*\d\w*'
    # Replace these words with an empty string
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text

def convert_numbers_to_words(text):
    # Function to replace all numbers with their word equivalents
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

    # Clean text by removing unwanted characters and whitespace
    # This pattern also includes the removal of newlines (\n) and tabs (\t)
    text = re.sub(r'[^\x00-\x7F]+|[^\w\s.]|\n|\t', ' ', text)

    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip().lower()

    sentences = [sentence for sentence in re.split(r'(?<=[.!?]) +', text) if 10 <= len(sentence.split()) <= 100]

    sentences = [s.replace('.', '').strip() for s in sentences]
    
    return {'sentences': sentences}

splits = ['validation', 'train']
datasets = load_dataset("wikitext", "wikitext-2-v1", split=splits) # 'train',
datasets = {split: ds for split, ds in zip(splits, datasets)}

processed_datasets = {split: ds.map(preprocess_text, batched=False, remove_columns=['text']) for split, ds in datasets.items()}

for split in processed_datasets:
    processed_datasets[split] = processed_datasets[split].map(lambda x: {'sentences': sum(x['sentences'], [])},
                                                              batched=True, batch_size=1000, num_proc=32)

def create_subset(dataset, subset_ratio=0.1, seed=42):
    shuffled_dataset = dataset.shuffle(seed=seed)

    num_samples = int(len(shuffled_dataset) * subset_ratio)

    subset = shuffled_dataset.select(range(num_samples))

    return subset

for split in splits:
    processed_datasets[split] = create_subset(processed_datasets[split], subset_ratio=0.05)
print(datasets)

def phoneme_extraction(example):
    # print(example['sentences'])
    phonemes = g2p(example['sentences'])
    return {'phonemes': phonemes}

print(processed_datasets[splits[0]][:10])

for split in processed_datasets:
    processed_datasets[split] = processed_datasets[split].map(phoneme_extraction, batched=True, batch_size=1, num_proc=1)
    # processed_datasets[split] = processed_datasets[split].map(phoneme_extraction, batched=False, num_proc=1)

print(processed_datasets[splits[0]][:10])

for split in processed_datasets:
    processed_datasets[split].save_to_disk(f"./wikitext-processed-{split}")
