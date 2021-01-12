import typing as t
import random
import functools

import csv
import sentencepiece
import pandas as pd
import torch
import torch.nn as nn


InstType = t.Union[torch.Tensor, t.List[int]]


class IterableDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        datagen,
        tokenizer_en: sentencepiece.SentencePieceProcessor,
        tokenizer_fi: sentencepiece.SentencePieceProcessor,
        max_sentence_len: int = 256,
    ):
        super(IterableDataset, self).__init__()

        self.datagen = datagen

        self._seed = 0
        self._tokenizer_en = tokenizer_en
        self._tokenizer_fi = tokenizer_fi
        self._max_sentence_len = max_sentence_len

    def __iter__(self):
        def _iter():
            for batch in self.datagen:
                batch = self._process_batch(batch)

                random.seed(self._seed)
                random.shuffle(batch)
                self._seed = (self._seed + 1) % int(1e7)

                for inst in batch:
                    if len(inst) <= self._max_sentence_len:
                        yield inst

        return _iter()

    def __len__(self):
        return len(self.data)

    @classmethod
    def _process_sentence(
        cls,
        sentence: str,
        tokenizer: sentencepiece.SentencePieceProcessor,
        append_eos: bool,
        append_bos: bool,
    ) -> InstType:
        sentence_enc = tokenizer.encode(sentence)

        if append_eos:
            sentence_enc.append(tokenizer.eos_id())

        if append_bos:
            sentence_enc.append(tokenizer.bos_id())

        sentence_enc = torch.tensor(sentence_enc, dtype=torch.long)

        return sentence_enc

    def _process_batch(self, batch) -> t.List[t.Tuple[InstType, InstType]]:
        processed_batch = []  # type: t.List[t.Tuple[t.List[int], t.List[int]]]

        for i, (s_en, s_fi) in batch.iterrows():
            s_en_enc = self._process_sentence(
                s_en, self._tokenizer_en, append_eos=True, append_bos=False
            )
            s_fi_enc = self._process_sentence(
                s_fi, self._tokenizer_fi, append_eos=False, append_bos=True
            )

            processed_batch.append((s_fi_enc, s_en_enc))

        return processed_batch


def load_data_from_file(
    filepath: str,
    max_dataset_size: t.Optional[int] = None,
    chunksize: t.Optional[int] = None,
):
    datagen = pd.read_csv(
        filepath,
        sep="\t",
        nrows=max_dataset_size,
        index_col=None,
        header=None,
        names=["en", "fi"],
        chunksize=chunksize,
        quoting=csv.QUOTE_NONE,
    )
    return datagen


def pre_collate_fn(
    batch: t.Tuple[t.List[torch.Tensor], t.List[torch.Tensor]], pad_id: int
) -> t.Tuple[torch.Tensor, torch.Tensor, t.List[int], t.List[int]]:
    sent_source_batch, sent_target_batch = zip(*batch)

    sent_source_lens = list(map(len, sent_source_batch))
    sent_target_lens = list(map(len, sent_target_batch))

    batch_size = len(sent_source_batch)

    # Note: concatenate all data in order to pad both batches to the same size
    all_data = (*sent_source_batch, *sent_target_batch)

    # Note: cast a list of tensors to a single 2-D tensor AND pad to
    # the same length.
    all_data = nn.utils.rnn.pad_sequence(
        all_data,
        padding_value=pad_id,
        batch_first=True,
    )
    # Note: switch sequence length <-> batch dimension
    all_data = torch.transpose(all_data, 1, 0)

    # Note: separate the data into the original batches
    sent_source_batch = all_data[:, :batch_size]
    sent_target_batch = all_data[:, batch_size:]

    return sent_source_batch, sent_target_batch, sent_source_lens, sent_target_lens


def get_data_stream(
    filepath: str,
    tokenizer_en: sentencepiece.SentencePieceProcessor,
    tokenizer_fi: sentencepiece.SentencePieceProcessor,
    batch_size: int,
    max_dataset_size: t.Optional[int] = None,
    max_sentence_len: int = 256,
    chunksize: int = 4096,
):
    # Note: a big chunksize is better because it helps breaking possible
    # sequence bias since we will be shuffling only example within the
    # same data chunk.
    data = load_data_from_file(
        filepath,
        max_dataset_size=max_dataset_size,
        chunksize=chunksize,
    )

    dataset = IterableDataset(
        data, tokenizer_en, tokenizer_fi, max_sentence_len=max_sentence_len
    )

    collate_fn = functools.partial(pre_collate_fn, pad_id=tokenizer_en.pad_id())

    datagen = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate_fn
    )

    return datagen
