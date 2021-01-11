"""
TODO:
- model Evalutation
- Output decodification
"""
import typing as t
import random
import functools

import tqdm
import torch
import torch.nn as nn
import sentencepiece

import utils


class IterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, data: t.List[t.Tuple[utils.InstType, utils.InstType]]):
        super(IterableDataset, self).__init__()
        self.data = data
        self._seed = 0

    def __iter__(self):
        random.seed(self._seed)
        random.shuffle(self.data)
        self._seed += 1
        return iter(self.data)

    def __len__(self):
        return len(self.data)


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


class FiEnTranslator(nn.Module):
    def __init__(
        self,
        vocab_size_source: int,
        vocab_size_target: int,
        d_model: int,
        n_encoder_layers: int,
        n_decoder_layers: int,
        pad_id: int,
        num_heads: int = 1,
    ):
        super(FiEnTranslator, self).__init__()

        self.input_encoder = nn.Sequential(
            nn.Embedding(vocab_size_source, d_model),
            nn.LSTM(d_model, d_model, num_layers=n_encoder_layers),
        )

        self.pre_attention_decoder = nn.Sequential(
            nn.Embedding(vocab_size_target, d_model),
            nn.LSTM(d_model, d_model),
        )

        self.attention_layer = nn.MultiheadAttention(d_model, num_heads)

        self.decoder_lstm = nn.LSTM(d_model, d_model, num_layers=n_decoder_layers)

        self.dense = nn.Linear(d_model, vocab_size_target)

        self._pad_id = pad_id

    def forward(self, sent_source: torch.Tensor, sent_target: torch.Tensor):
        sent_source_enc, _ = self.input_encoder(sent_source)
        sent_target_enc, _ = self.pre_attention_decoder(sent_target)

        pad_mask = torch.transpose(sent_source == self._pad_id, 1, 0)

        out, _ = self.attention_layer(
            query=sent_target_enc,
            key=sent_source_enc,
            value=sent_source_enc,
            key_padding_mask=pad_mask,
        )

        # Note: skip connection
        out = out + sent_target_enc

        out, _ = self.decoder_lstm(out)
        out = self.dense(out)

        return out


def calc_acc(preds: torch.Tensor, true: torch.Tensor, pad_id: int) -> float:
    with torch.no_grad():
        # Note: select only non-pad tokens
        mask = true != pad_id

        total_correct = (
            torch.masked_select(preds.argmax(dim=1) == true, mask).sum().item()
        )
        total_valid_tokens = mask.sum().item()
        acc = total_correct / total_valid_tokens

    return acc


def _test():
    train_epochs = 10
    checkpoint_path = "checkpoint.pt"
    batch_size = 4
    device = "cuda"

    tokenizer_en = sentencepiece.SentencePieceProcessor(model_file="vocab/en_bpe.model")
    tokenizer_fi = sentencepiece.SentencePieceProcessor(model_file="vocab/fi_bpe.model")

    data_train, data_eval = utils.get_train_eval_data(
        tokenizer_en,
        tokenizer_fi,
        max_dataset_size=1024,
        chunksize=512,
        device=device,
    )

    dataset = IterableDataset(data_train)
    collate_fn = functools.partial(pre_collate_fn, pad_id=tokenizer_en.pad_id())

    train_datagen = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate_fn
    )

    vocab_size_source = tokenizer_fi.vocab_size()
    vocab_size_target = tokenizer_en.vocab_size()

    model = FiEnTranslator(
        vocab_size_source=vocab_size_source,
        vocab_size_target=vocab_size_target,
        d_model=1024,
        n_encoder_layers=2,
        n_decoder_layers=2,
        pad_id=tokenizer_en.pad_id(),
        num_heads=4,
    )
    optim = torch.optim.Adam(model.parameters(), 0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer_fi.pad_id())

    try:
        model.load_state_dict(torch.load(checkpoint_path))
        print("Loaded checkpoint file.")

    except FileNotFoundError:
        pass

    model = model.to(device)

    epoch_batches_num = (len(dataset) + batch_size - 1) // batch_size

    for epoch in range(1, 1 + train_epochs):
        print(f"Epoch: {epoch} / {train_epochs} ...")

        train_acc = 0.0
        model.train()

        for (
            sent_source_batch,
            sent_target_batch,
            sent_source_lens,
            sent_target_lens,
        ) in tqdm.auto.tqdm(train_datagen, total=epoch_batches_num):
            optim.zero_grad()

            sent_target_preds = model(sent_source_batch, sent_target_batch)

            sent_target_preds = sent_target_preds.view(-1, vocab_size_target)
            sent_target_batch = sent_target_batch.reshape(-1)

            loss = criterion(sent_target_preds, sent_target_batch)
            loss.backward()
            optim.step()

            train_acc += calc_acc(
                sent_target_preds, sent_target_batch, tokenizer_fi.pad_id()
            )

        train_acc /= epoch_batches_num

        model.eval()
        print(f"Train acc : {train_acc:.4f}")
        # print(f"Eval acc  : {eval_acc:.4f}")

    # torch.save(model.state_dict(), checkpoint_path)
    print("Done.")


if __name__ == "__main__":
    _test()
