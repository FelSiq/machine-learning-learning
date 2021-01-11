import sentencepiece
import pandas as pd

import utils


if __name__ == "__main__":
    datagen = utils.load_data_from_file(int(4e6), None)

    datagen["en"].to_csv("corpus/en-only.txt", index=False, header=False)
    datagen["fi"].to_csv("corpus/fi-only.txt", index=False, header=False)

    del datagen

    sentencepiece.SentencePieceTrainer.train(
        input="corpus/en-only.txt",
        model_prefix="vocab/en_bpe",
        vocab_size=32000,
        shuffle_input_sentence=True,
        pad_id=3,
    )

    sentencepiece.SentencePieceTrainer.train(
        input="corpus/fi-only.txt",
        model_prefix="vocab/fi_bpe",
        vocab_size=32000,
        shuffle_input_sentence=True,
        pad_id=3,
    )
