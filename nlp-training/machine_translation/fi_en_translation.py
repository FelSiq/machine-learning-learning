"""
TODO:
- Model evalutation
- Output decodification
- Set up a learning rate warm-up + decay
- Train with more data for more epochs and see what happens
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

        self.input_encoder = nn.Sequential(
            nn.Embedding(vocab_size_source, d_model),
            nn.LSTM(d_model, d_model, num_layers=n_encoder_layers),
        )

        self.pre_attention_decoder = nn.Sequential(
            nn.Embedding(vocab_size_target, d_model),
            nn.LSTM(d_model, d_model),
        )

        self.attention_layer = nn.MultiheadAttention(d_model, n_heads)

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
                torch.masked_select(preds.argmax(dim=1) == true, mask).sum().item()
            )
            total_valid_tokens = mask.sum().item()
            acc = total_correct / total_valid_tokens

        else:
            acc = (preds.argmax(dim=-1) == true).float().mean().item()

    return acc


def calc_loss(model, sent_source_batch, sent_target_batch, criterion, device):
    sent_source_batch = sent_source_batch.to(device)
    sent_target_batch = sent_target_batch.to(device)

    sent_target_preds = model(sent_source_batch, sent_target_batch)

    # Note: shift to the left here in order to remove BOS.
    sent_target_batch = sent_target_batch[1:, ...]
    sent_target_preds = sent_target_preds[:-1, ...]

    vocab_size_target = sent_target_preds.shape[-1]
    sent_target_preds = sent_target_preds.view(-1, vocab_size_target)
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
            model, sent_source_batch, sent_target_batch, criterion, device
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
            model, sent_source_batch, sent_target_batch, criterion, device
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


def _test():
    train_epochs = 20
    checkpoint_path = "checkpoint.pt"
    device = "cuda"
    load_checkpoint = False
    epochs_per_checkpoint = 5
    ignore_pad_id = False

    max_sentence_len_train = 256
    max_sentence_len_eval = 256
    d_model = 1024
    n_decoder_layers = 2
    n_encoder_layers = 2
    n_heads = 4

    batch_size_train = 4
    batch_size_eval = 4
    # train_size = 7196119
    # eval_size = 72689
    train_size = 2000
    eval_size = 100

    tokenizer_en = sentencepiece.SentencePieceProcessor(
        model_file="./vocab/en_bpe.model"
    )
    tokenizer_fi = sentencepiece.SentencePieceProcessor(
        model_file="./vocab/fi_bpe.model"
    )

    vocab_size_source = tokenizer_fi.vocab_size()
    vocab_size_target = tokenizer_en.vocab_size()

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

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim,
        factor=0.5,
        min_lr=0.0001,
        verbose=True,
    )

    if load_checkpoint:
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            print("Loaded checkpoint file.")

            model.load_state_dict(checkpoint["model"])
            model.to(device)
            optim.load_state_dict(checkpoint["optim"])

        except FileNotFoundError:
            pass

    else:
        model = model.to(device)

    if ignore_pad_id:
        criterion = nn.CrossEntropyLoss()

    else:
        criterion = nn.CrossEntropyLoss(ignore_index=tokenizer_fi.pad_id())

    epoch_batches_num_train = (train_size + batch_size_train - 1) // batch_size_train
    epoch_batches_num_eval = (eval_size + batch_size_eval - 1) // batch_size_eval

    full_datagen_train = lambda: tqdm.auto.tqdm(
        datagen_train(), total=epoch_batches_num_train
    )
    full_datagen_eval = lambda: tqdm.auto.tqdm(
        datagen_eval(), total=epoch_batches_num_eval
    )

    for epoch in range(1, 1 + train_epochs):
        print(f"Epoch: {epoch} / {train_epochs} ...")

        acc_train, loss_train = run_train_epoch(
            model,
            optim,
            criterion,
            device,
            full_datagen_train(),
            pad_id=tokenizer_en.pad_id(),
            ignore_pad_id=ignore_pad_id,
        )
        acc_eval, loss_eval = run_eval_epoch(
            model,
            scheduler,
            criterion,
            device,
            full_datagen_eval(),
            pad_id=tokenizer_en.pad_id(),
            ignore_pad_id=ignore_pad_id,
        )

        print(f"Train loss : {loss_train:.4f} - Train acc : {acc_train:.4f}")
        print(f"Eval loss  : {loss_eval:.4f} - EVal acc  : {acc_eval:.4f}")

        if epochs_per_checkpoint > 0 and (
            epoch % epochs_per_checkpoint == 0 or epoch == train_epochs
        ):
            print("Saving checkpoint...")
            checkpoint = {
                "model": model.state_dict(),
                "optim": optim.state_dict(),
            }
            torch.save(checkpoint, checkpoint_path)
            print("Done.")


if __name__ == "__main__":
    _test()
