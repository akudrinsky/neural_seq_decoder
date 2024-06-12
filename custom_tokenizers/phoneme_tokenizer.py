
class PhonemeTokenizer():
    def __init__(self):
        super().__init__()
        token_list = ['<PAD>', '<BOS>', '<EOS>', '<UNK>', ' ',
            'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY',
            'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P',
            'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH'
        ]
        self.add_tokens(token_list)
        self.pad_token = '<PAD>'
        self.bos_token = '<BOS>'
        self.eos_token = '<EOS>'
        self.unk_token = '<UNK>'
        self.space_token = ' '

        self.token_to_id = {token: idx for idx, token in enumerate(token_list)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    def encode(self, phonemes):
        return [self.token_to_id(token) for token in phonemes]

    def decode(self, ids):
        return [self.id_to_token(idx) for idx in ids]
