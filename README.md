
To train brain encoder:
1. Start with dataset downloading and preprocessing
    - Download data from https://datadryad.org/stash/dataset/doi:10.5061/dryad.x69p8czpq (https://datadryad.org/stash/downloads/file_stream/2547369)
    - Then use preprocess_data/preprocess_competition_dataset.py
    - Then add phonemes using preprocess_data/add_phonemes.py script
2. TODO...


To train phoneme2phoneme gpt2 model:
1. Download librispeech corpus
2. Use train_p2p.py script
