"""
TODO:
- Output decodification
"""
import typing as t
import functools
import math

import tqdm
import torch
import torch.nn as nn
import sentencepiece

import utils


class FiEnTranslatorRNN(nn.Module):
    def __init__(
        self,
        vocab_size_source: int,
        vocab_size_target: int,
        d_model: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        n_heads: int,
        pad_id: int,
        bidirectional: bool = True,
    ):
        super(FiEnTranslatorRNN, self).__init__()

        self.embedding_source = nn.Embedding(vocab_size_source, d_model)
        self.input_encoder_rnn = nn.LSTM(
            d_model, d_model, num_layers=num_encoder_layers, bidirectional=bidirectional
        )

        self.embedding_target = nn.Embedding(vocab_size_target, d_model)
        self.pre_attention_decoder_rnn = nn.LSTM(
            d_model, d_model, bidirectional=bidirectional
        )

        d_model_bi = (1 + int(bidirectional)) * d_model

        self.attention_layer = nn.MultiheadAttention(d_model_bi, n_heads)

        self.decoder_lstm = nn.LSTM(
            d_model_bi, d_model_bi, num_layers=num_decoder_layers
        )

        self.dense = nn.Linear(d_model_bi, vocab_size_target)

        self._pad_id = pad_id

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        len_x: t.Sequence[int],
        len_y: t.Sequence[int],
    ):
        _total_length = max(max(len_x), max(len_y))

        x = self.embedding_source(x)
        x = nn.utils.rnn.pack_padded_sequence(x, len_x, enforce_sorted=False)
        x, _ = self.input_encoder_rnn(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(
            x,
            padding_value=self._pad_id,
            total_length=_total_length,
        )

        y = self.embedding_target(y)
        y = nn.utils.rnn.pack_padded_sequence(y, len_y, enforce_sorted=False)
        y, _ = self.pre_attention_decoder_rnn(y)
        y, _ = nn.utils.rnn.pad_packed_sequence(
            y,
            padding_value=self._pad_id,
            total_length=_total_length,
        )

        out, _ = self.attention_layer(
            query=y,
            key=x,
            value=x,
        )

        # Note: skip connection
        out = out + y

        out, _ = self.decoder_lstm(out)
        out = self.dense(out)

        return out


class PositionalEncoding(nn.Module):
    # Source: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, d_model, dropout=0.0, max_len=30):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(100.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class FiEnTranslatorTransformer(nn.Module):
    def __init__(
        self,
        n_in_vocab: int,
        n_out_vocab: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dropout: float,
        device: str,
    ):
        super(FiEnTranslatorTransformer, self).__init__()

        self.model_type = "Transformer"

        self.input_X = nn.Sequential(
            nn.Embedding(n_in_vocab, d_model),
            PositionalEncoding(d_model, dropout=dropout, max_len=d_model),
        )

        self.input_Y = nn.Sequential(
            nn.Embedding(n_in_vocab, d_model),
            PositionalEncoding(d_model, dropout=dropout, max_len=d_model),
        )

        self.transformer = nn.Transformer(
            d_model,
            nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
        )

        self.linear = nn.Linear(d_model, n_out_vocab)

        self.device = device

    def forward(self, X, Y, x_len=None, y_len=None):
        y_len = Y.shape[0]

        _mask = torch.triu(
            torch.full((y_len, y_len), float("-inf"), device=self.device), diagonal=1
        )

        X = self.input_X(X)
        Y = self.input_Y(Y)
        X = self.transformer(X, Y, tgt_mask=_mask)
        X = self.linear(X)
        return X


def calc_acc(
    preds: torch.Tensor,
    true: torch.Tensor,
    pad_id: t.Optional[int],
    ignore_pad_id: bool,
) -> float:
    with torch.no_grad():
        if ignore_pad_id:
            if pad_id is None:
                raise TypeError("'ignore_pad_ind' is True but 'pad_id' is None.")

            # Note: select only non-pad tokens
            mask = true != pad_id

            preds = preds.argmax(dim=-1)

            total_correct = torch.masked_select(preds == true, mask).sum().item()

            total_valid_tokens = mask.sum().item()
            acc = total_correct / total_valid_tokens

        else:
            acc = (preds.argmax(dim=-1) == true).float().mean().item()

    return acc


def calc_loss(
    model,
    sent_source_batch,
    sent_target_batch,
    sent_source_lens,
    sent_target_lens,
    criterion,
    device,
):
    sent_source_batch = sent_source_batch.to(device)
    sent_target_batch = sent_target_batch.to(device)

    sent_target_preds = model(
        sent_source_batch, sent_target_batch, sent_source_lens, sent_target_lens
    )

    # Note: shift to the left here in order to remove BOS.
    sent_target_batch = sent_target_batch[1:, ...]
    sent_target_preds = sent_target_preds[:-1, ...]

    # Note: switch batch_dim <-> sequence_dim
    sent_target_batch = torch.transpose(sent_target_batch, 0, 1)
    sent_target_preds = torch.transpose(sent_target_preds, 0, 1)

    vocab_size_target = sent_target_preds.shape[-1]
    sent_target_preds = sent_target_preds.reshape(-1, vocab_size_target)
    sent_target_batch = sent_target_batch.reshape(-1)

    loss = criterion(sent_target_preds, sent_target_batch)

    return loss, sent_target_preds, sent_target_batch


def run_train_epoch(
    model,
    optim,
    criterion,
    device,
    datagen_train,
    pad_id: t.Optional[int],
    ignore_pad_id: bool,
):
    train_acc = total_train_loss = 0.0
    total_epochs = 0

    model.train()

    for (
        sent_source_batch,
        sent_target_batch,
        sent_source_lens,
        sent_target_lens,
    ) in datagen_train:
        optim.zero_grad()

        loss, sent_target_preds, sent_target_batch = calc_loss(
            model,
            sent_source_batch,
            sent_target_batch,
            sent_source_lens,
            sent_target_lens,
            criterion,
            device,
        )

        loss.backward()
        optim.step()

        train_acc += calc_acc(
            sent_target_preds,
            sent_target_batch,
            pad_id=pad_id,
            ignore_pad_id=ignore_pad_id,
        )
        total_train_loss += loss.item()

        del sent_source_batch
        del sent_target_batch
        del sent_target_preds

        total_epochs += 1

    train_acc /= total_epochs
    total_train_loss /= total_epochs

    return train_acc, total_train_loss


def run_eval_epoch(
    model,
    scheduler,
    criterion,
    device,
    datagen_eval,
    pad_id: t.Optional[int],
    ignore_pad_id: bool,
):
    eval_acc = total_eval_loss = 0.0
    total_epochs = 0

    model.eval()

    for (
        sent_source_batch,
        sent_target_batch,
        sent_source_lens,
        sent_target_lens,
    ) in datagen_eval:
        loss, sent_target_preds, sent_target_batch = calc_loss(
            model,
            sent_source_batch,
            sent_target_batch,
            sent_source_lens,
            sent_target_lens,
            criterion,
            device,
        )

        eval_acc += calc_acc(
            sent_target_preds,
            sent_target_batch,
            pad_id=pad_id,
            ignore_pad_id=ignore_pad_id,
        )
        total_eval_loss += loss.item()

        del sent_source_batch
        del sent_target_batch
        del sent_target_preds

        total_epochs += 1

    eval_acc /= total_epochs
    total_eval_loss /= total_epochs

    scheduler.step(total_eval_loss)

    return eval_acc, total_eval_loss


def get_data_streams(
    tokenizer_en: sentencepiece.SentencePieceProcessor,
    tokenizer_fi: sentencepiece.SentencePieceProcessor,
):
    max_sentence_len_train = 256
    max_sentence_len_eval = 256
    batch_size_train = 4
    batch_size_eval = 4
    train_size = min(1024, 7196119)
    eval_size = min(128, 72689)

    datagen_train = functools.partial(
        utils.get_data_stream,
        filepath="./corpus/en-fi-train.txt",
        tokenizer_en=tokenizer_en,
        tokenizer_fi=tokenizer_fi,
        batch_size=batch_size_train,
        max_dataset_size=train_size,
        max_sentence_len=max_sentence_len_train,
    )
    datagen_eval = functools.partial(
        utils.get_data_stream,
        filepath="./corpus/en-fi-eval.txt",
        tokenizer_en=tokenizer_en,
        tokenizer_fi=tokenizer_fi,
        batch_size=batch_size_eval,
        max_dataset_size=eval_size,
        max_sentence_len=max_sentence_len_eval,
    )

    epoch_batches_num_train = (train_size + batch_size_train - 1) // batch_size_train
    epoch_batches_num_eval = (eval_size + batch_size_eval - 1) // batch_size_eval

    full_datagen_train = lambda: tqdm.auto.tqdm(
        datagen_train(), total=epoch_batches_num_train
    )
    full_datagen_eval = lambda: tqdm.auto.tqdm(
        datagen_eval(), total=epoch_batches_num_eval
    )

    return full_datagen_train, full_datagen_eval


def get_model_optim_scheduler(
    tokenizer_en: sentencepiece.SentencePieceProcessor,
    tokenizer_fi: sentencepiece.SentencePieceProcessor,
    device: str,
    load_checkpoint: bool,
    checkpoint_path: str,
) -> t.Tuple[
    nn.Module, torch.optim.Adam, torch.optim.lr_scheduler.ReduceLROnPlateau, int
]:
    d_model = 512
    bidirectional = True
    num_decoder_layers = 4
    num_encoder_layers = 4
    n_heads = 8
    dropout = 0.1

    scheduler_patience = 5
    scheduler_ratio = 0.5

    vocab_size_source = tokenizer_fi.vocab_size()
    vocab_size_target = tokenizer_en.vocab_size()

    """
    # Note: unused, much less eficient than the transformer one
    model = FiEnTranslatorRNN(
        vocab_size_source=vocab_size_source,
        vocab_size_target=vocab_size_target,
        d_model=d_model,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        pad_id=tokenizer_en.pad_id(),
        n_heads=n_heads,
        bidirectional=bidirectional,
    )
    """

    model = FiEnTranslatorTransformer(
        n_in_vocab=vocab_size_source,
        n_out_vocab=vocab_size_target,
        d_model=256,  # Note: d_model = max sentence len
        nhead=n_heads,
        dropout=dropout,
        dim_feedforward=d_model,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        device=device,
    )

    optim = torch.optim.Adam(model.parameters(), 1e-4)

    start_epoch = 0

    if load_checkpoint:
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            print("Loaded checkpoint file.")

            model.load_state_dict(checkpoint["model"])
            model = model.to(device)
            optim.load_state_dict(checkpoint["optim"])
            start_epoch = checkpoint["epoch"]

        except FileNotFoundError:
            pass

    model = model.to(device)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim,
        factor=scheduler_ratio,
        patience=scheduler_patience,
        min_lr=0.0001,
        verbose=True,
    )

    return model, optim, scheduler, start_epoch


def predict(model, sentence, device, tokenizer_fi, tokenizer_en):
    model.eval()
    prepared = torch.tensor(
        tokenizer_fi.encode(sentence), dtype=torch.long, device=device
    ).unsqueeze(0)

    prepared = prepared.to(device)
    out = torch.zeros((1, 256), device=device, dtype=torch.long)
    out[0][0] = tokenizer_fi.pad_id()

    out = torch.transpose(out, 0, 1)
    prepared = torch.transpose(prepared, 0, 1)

    i = 0
    last_token = None
    eos_id = tokenizer_en.eos_id()

    while i < 256 and last_token != eos_id:
        pred = model(prepared, out)
        ind = pred[i - 1].squeeze().argmax(dim=-1)
        last_token = ind.item()
        out[i][0] = last_token
        i += 1

    out = out.squeeze().detach().cpu().numpy().tolist()

    print("Test input sentence (finnish) :", sentence)
    print("Model's output      (english) :", tokenizer_en.decode(out))


def _test():
    train_epochs = 1
    checkpoint_path = "./checkpoint.tar"
    device = "cuda"
    load_checkpoint = True
    epochs_per_checkpoint = 0
    ignore_pad_id = True

    tokenizer_en = sentencepiece.SentencePieceProcessor(
        model_file="./vocab/en_bpe.model"
    )
    tokenizer_fi = sentencepiece.SentencePieceProcessor(
        model_file="./vocab/fi_bpe.model"
    )

    model, optim, scheduler, start_epoch = get_model_optim_scheduler(
        tokenizer_en,
        tokenizer_fi,
        device,
        load_checkpoint,
        checkpoint_path,
    )

    if ignore_pad_id:
        criterion = nn.CrossEntropyLoss()

    else:
        criterion = nn.CrossEntropyLoss(
            ignore_index=tokenizer_fi.pad_id(), size_average=True
        )

    datagen_train, datagen_eval = get_data_streams(tokenizer_en, tokenizer_fi)

    for epoch in range(1 + start_epoch, 1 + start_epoch + train_epochs):
        print(f"Epoch: {epoch} / {start_epoch + train_epochs} ...")

        acc_train, loss_train = run_train_epoch(
            model,
            optim,
            criterion,
            device,
            datagen_train(),
            pad_id=tokenizer_en.pad_id(),
            ignore_pad_id=ignore_pad_id,
        )
        acc_eval, loss_eval = run_eval_epoch(
            model,
            scheduler,
            criterion,
            device,
            datagen_eval(),
            pad_id=tokenizer_en.pad_id(),
            ignore_pad_id=ignore_pad_id,
        )

        print(f"Train loss : {loss_train:.4f} - Train acc : {acc_train:.4f}")
        print(f"Eval loss  : {loss_eval:.4f} - Eval acc  : {acc_eval:.4f}")

        if epochs_per_checkpoint > 0 and (
            epoch % epochs_per_checkpoint == 0 or epoch == train_epochs
        ):
            print("Saving checkpoint...")
            checkpoint = {
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, checkpoint_path)
            print("Done.")

    test_inp = "Olen tehnyt sen!"
    predict(model, test_inp, device, tokenizer_fi, tokenizer_en)


if __name__ == "__main__":
    _test()
