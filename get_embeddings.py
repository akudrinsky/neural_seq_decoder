import torch
import pickle
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Model

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = GPT2Model.from_pretrained("openai-community/gpt2").cuda()
model.eval()

with open("/mnt/scratch/kudrinsk/eval_challenge/initial_dataset.pickle", "rb") as handle:
    data = pickle.load(handle)

def get_text_embedding(model, tokenizer, text):
    input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        outputs = model(input_ids)
    return outputs.last_hidden_state[0, -1, :]

for sample in tqdm(data):
    text = sample['text']
    embedding = get_text_embedding(model, tokenizer, text)
    sample['embedding'] = embedding

with open("/mnt/scratch/kudrinsk/eval_challenge/gpt2_text_embeds.pickle", "wb") as handle:
    pickle.dump(data, handle)
