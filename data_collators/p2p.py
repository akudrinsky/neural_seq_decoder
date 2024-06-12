import torch
from torch.nn.utils.rnn import pad_sequence


class P2PDataCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        input_ids = [torch.tensor(feature['input_ids']) for feature in features]
        labels = [torch.tensor(feature['labels']) for feature in features]

        # Pad input_ids and labels
        padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        # Create attention masks for the inputs
        attention_mask = (padded_input_ids != self.pad_token_id).long()

        return {
            'input_ids': padded_input_ids,
            'attention_mask': attention_mask,
            'labels': padded_labels
        }