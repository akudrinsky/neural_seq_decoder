from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
import torch
import string

# Define the phoneme list and tokens
def get_phoneme_list():
    phonemes_list = [
        'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 
        'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 
        'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH', 'spn'
    ]
    return phonemes_list

tokens = ['<PAD>', '<EOS>', '<START_DEC>', '<UNK>', ' '] + get_phoneme_list()
token_set = set(tokens)

token_to_index = {token: idx for idx, token in enumerate(tokens)}
index_to_token = {idx: token for token, idx in token_to_index.items()}

# Load the trained model
config = T5Config(
    vocab_size=len(tokens),
    num_layers=6,
    pad_token_id=token_to_index['<PAD>'], 
    eos_token_id=token_to_index['<EOS>'],
    decoder_start_token_id=token_to_index['<START_DEC>'],
)
model = T5ForConditionalGeneration.from_pretrained('./p2p_modify_time_augs//checkpoint-403500/')
model = model.cuda()  # Move the model to GPU if available
model.eval()  # Set the model to evaluation mode

# Inference function
def phonemes_to_sentence(phonemes):
    # Convert phonemes to input IDs
    input_ids = [token_to_index[ph] for ph in phonemes if ph in token_to_index]
    input_ids = torch.tensor(input_ids).unsqueeze(0).cuda()  # Add batch dimension and move to GPU

    # Generate sentence
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_length=100,
            num_beams=5,
            early_stopping=True,
            length_penalty=1.0,  # You can adjust this to penalize longer sequences
            no_repeat_ngram_size=2  # Ensure no repeated 2-grams
        )
    generated_ids = outputs[0].tolist()
    if token_to_index['<EOS>'] in generated_ids:
        print('!')
        eos_index = generated_ids.index(token_to_index['<EOS>'])
        generated_ids = generated_ids[:eos_index]

    sentence = '-'.join([index_to_token[idx] for idx in generated_ids if idx in index_to_token])

    # Decode the generated sentence
    # sentence = ''.join([index_to_token[idx.item()] for idx in outputs[0] if idx.item() in index_to_token])

    return sentence

# Example usage
phoneme_list = ['B', 'IH', 'G', ' ', 'S', 'M', 'L', ' ', 'AE', 'P', 'L', ' ', 'CH', 'EH', 'K', '<EOS>']
sentence = phonemes_to_sentence(phoneme_list)
print(f"Generated Sentence: {sentence}")
