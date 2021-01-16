"""
TODO:
- Output decodification
"""
import typing as t
import functools

import tqdm
import torch
import torch.nn as nn
import sentencepiece

import utils


class FiEnTranslator(nn.Module):
    def __init__(
        self,
        vocab_size_source: int,
        vocab_size_target: int,
        d_model: int,
        n_encoder_layers: int,
        n_decoder_layers: int,
        n_heads: int,
        pad_id: int,
    ):
        super(FiEnTranslator, self).__init__()

        self.embedding_source = nn.Embedding(vocab_size_source, d_model)
        self.input_encoder_rnn = nn.LSTM(d_model, d_model, num_layers=n_encoder_layers)

        self.embedding_target = nn.Embedding(vocab_size_target, d_model)
        self.pre_attention_decoder_rnn = nn.LSTM(d_model, d_model)

        self.attention_layer = nn.MultiheadAttention(d_model, n_heads)

        self.decoder_lstm = nn.LSTM(d_model, d_model, num_layers=n_decoder_layers)

        self.dense = nn.Linear(d_model, vocab_size_target)

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

            total_correct = (
                torch.masked_select(preds.argmax(dim=-1) == true, mask).sum().item()
            )
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
    # train_size = 7196119
    # eval_size = 72689
    train_size = 4
    eval_size = 4

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
    d_model = 1024
    n_decoder_layers = 2
    n_encoder_layers = 2
    n_heads = 4

    scheduler_patience = 5
    scheduler_ratio = 0.5

    vocab_size_source = tokenizer_fi.vocab_size()
    vocab_size_target = tokenizer_en.vocab_size()

    model = FiEnTranslator(
        vocab_size_source=vocab_size_source,
        vocab_size_target=vocab_size_target,
        d_model=d_model,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
        pad_id=tokenizer_en.pad_id(),
        n_heads=n_heads,
    )

    optim = torch.optim.Adam(model.parameters(), 0.01)

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


def _test():
    train_epochs = 500
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


if __name__ == "__main__":
    _test()
