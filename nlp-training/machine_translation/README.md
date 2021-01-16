# Finnish to English Neural Machine Translation
Translate finnish sentences to english using Neural Machine Translation.

## Install instructions
1. Install python library dependencies:
```bash
pip install -Ur requirements.txt
```

2. Get the train and evaluation corpus and the Byte Pair Encoding (BPE) vocabulary:
```bash
./get_data.sh
```

3. Start training:
```bash
python fi_en_translation.py
```
**Note:** you may want to open `fi_en_translation.py` and change some hyperparameters related to both the model and the training/evaluation data streams.
