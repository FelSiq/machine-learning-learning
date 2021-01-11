import typing as t
import random

import csv
import sentencepiece
import pandas as pd
import torch


InstType = t.Union[torch.Tensor, t.List[int]]


def load_data_from_file(
    max_dataset_size: t.Optional[int] = None, chunksize: t.Optional[int] = None
):
    datagen = pd.read_csv(
        "corpus/en-fi.txt",
        sep="\t",
        nrows=max_dataset_size,
        index_col=None,
        header=None,
        names=["en", "fi"],
        chunksize=chunksize,
        quoting=csv.QUOTE_NONE,
    )
    return datagen


def process_sentence(
    sentence: str,
    tokenizer: sentencepiece.SentencePieceProcessor,
    as_tensor: bool = False,
    device: str = "cpu",
) -> InstType:
    sentence_enc = tokenizer.encode(sentence)
    sentence_enc.append(tokenizer.eos_id())

    if as_tensor:
        sentence_enc = torch.tensor(sentence_enc, dtype=torch.long, device=device)

    return sentence_enc


def get_data(
    tokenizer_en: sentencepiece.SentencePieceProcessor,
    tokenizer_fi: sentencepiece.SentencePieceProcessor,
    max_dataset_size: t.Optional[int] = None,
    chunksize: t.Optional[int] = None,
    as_tensor: bool = False,
    device: str = "cpu",
    finnish_first: bool = True,
) -> t.List[t.Tuple[InstType, InstType]]:
    datagen = load_data_from_file(max_dataset_size, chunksize)

    data = []  # type: t.List[t.Tuple[t.List[int], t.List[int]]]

    for block in datagen:
        for i, (s_en, s_fi) in block.iterrows():
            s_en_enc = process_sentence(s_en, tokenizer_en, as_tensor, device)
            s_fi_enc = process_sentence(s_fi, tokenizer_fi, as_tensor, device)

            if finnish_first:
                data.append((s_fi_enc, s_en_enc))

            else:
                data.append((s_en_enc, s_fi_enc))

    return data


def filter_by_length(sequences: t.List[t.Tuple[InstType, InstType]], max_seq_len: int):
    i = 0
    while i < len(sequences):
        seq_a, seq_b = sequences[i]

        if max(len(seq_a), len(seq_b)) > max_seq_len:
            sequences.pop(i)

        else:
            i += 1


def get_train_eval_data(
    tokenizer_en: sentencepiece.SentencePieceProcessor,
    tokenizer_fi: sentencepiece.SentencePieceProcessor,
    max_dataset_size: t.Optional[int] = None,
    chunksize: t.Optional[int] = None,
    train_frac: float = 0.99,
    max_seq_len_train: int = 256,
    max_seq_len_eval: int = 256,
    random_seed: int = 16,
    as_tensor: bool = True,
    device: str = "cpu",
    verbose: bool = True,
) -> t.List[t.Tuple[InstType, InstType]]:
    assert 0.0 <= train_frac <= 1.0

    data = get_data(
        tokenizer_en,
        tokenizer_fi,
        as_tensor=as_tensor,
        max_dataset_size=max_dataset_size,
        chunksize=chunksize,
        device=device)

    random.seed(random_seed)
    random.shuffle(data)

    train_size = int(train_frac * len(data))

    data_train = data[:train_size]
    data_eval = data[train_size:]

    filter_by_length(data_train, max_seq_len_train)
    filter_by_length(data_eval, max_seq_len_eval)

    if verbose:
        print("Data information:")
        print("Data type  :", type(data_train[0]))
        print("Train size :", len(data_train))
        print("Eval size  :", len(data_eval))

    del data

    return data_train, data_eval
