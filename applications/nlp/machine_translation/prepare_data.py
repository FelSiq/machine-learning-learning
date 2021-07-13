import os

import csv
import sentencepiece
import pandas as pd

import utils


def split_train_eval_sets(train_frac: float = 0.99):
    data = utils.load_data_from_file("./corpus/en-fi.txt", None, None)

    train_size = int(train_frac * len(data))

    print("Train fraction :", train_frac)
    print("Train size     :", train_size)
    print("Eval size      :", len(data) - train_size)

    data.iloc[:train_size, :].to_csv(
        "./corpus/en-fi-train.txt",
        index=False,
        header=False,
        quoting=csv.QUOTE_NONE,
        quotechar="",
        sep="\t",
    )
    data.iloc[train_size:, :].to_csv(
        "./corpus/en-fi-eval.txt",
        index=False,
        header=False,
        quoting=csv.QUOTE_NONE,
        quotechar="",
        sep="\t",
    )

    del data
    del train_size


def create_byte_pair_encoding_vocab(max_sentences: int = int(3.5e6)):
    datagen = utils.load_data_from_file("./corpus/en-fi-train.txt", max_sentences, None)

    datagen["en"].to_csv("./corpus/en-only.txt", index=False, header=False)
    datagen["fi"].to_csv("./corpus/fi-only.txt", index=False, header=False)

    del datagen

    sentencepiece.SentencePieceTrainer.train(
        input="./corpus/en-only.txt",
        model_prefix="./vocab/en_bpe",
        vocab_size=32000,
        shuffle_input_sentence=True,
        pad_id=3,
    )

    sentencepiece.SentencePieceTrainer.train(
        input="./corpus/fi-only.txt",
        model_prefix="./vocab/fi_bpe",
        vocab_size=32000,
        shuffle_input_sentence=True,
        pad_id=3,
    )


if __name__ == "__main__":
    if os.path.isfile("./corpus/en-fi-train.txt") and os.path.isfile(
        "./corpus/en-fi-eval.txt"
    ):
        print("Train/eval files found, skipping data split.")

    else:
        print("Splitting data into train and evaluation files...")
        split_train_eval_sets()
        print("Done.")

    try:
        os.mkdir("./vocab")

    except FileExistsError:
        print("Vocabulary directory found.")
        pass

    if os.path.isfile("./vocab/fi_bpe.model") and os.path.isfile("vocab/en_bpe.model"):
        print("Byte Pair Encoding vocabulary files found, skipping this step.")

    else:
        print("Creating Byte Pair Encoding vocabulary using train files...")
        create_byte_pair_encoding_vocab()
        print("Done.")
