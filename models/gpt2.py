from transformers import GPT2LMHeadModel, GPT2Config

def get_phoneme_gpt2_model(vocab_size, pad_token_id, eos_token_id):
    config = GPT2Config.from_pretrained('gpt2')

    config.vocab_size = vocab_size
    config.pad_token_id = pad_token_id
    config.eos_token_id = eos_token_id
    model = GPT2LMHeadModel.from_pretrained('gpt2', config=config, ignore_mismatched_sizes=True)

    return model
