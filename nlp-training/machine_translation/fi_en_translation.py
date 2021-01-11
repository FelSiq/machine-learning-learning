"""
TODO:
- Model evalutation
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
    train_epochs = 2
    epochs_per_checkpoint = 1
    checkpoint_path = "checkpoint.pt"
    batch_size_train = 4
    batch_size_eval = 4
    device = "cuda"
    max_sentence_len_train = 256
    max_sentence_len_eval = 256

    train_size = 7196119
    eval_size = 72689
    train_size = 10000
    eval_size = 32

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

    epoch_batches_num = (train_size + batch_size_train - 1) // batch_size_train

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

    for epoch in range(1, 1 + train_epochs):
        print(f"Epoch: {epoch} / {train_epochs} ...")

        train_acc = 0.0
        model.train()

        for (
            sent_source_batch,
            sent_target_batch,
            sent_source_lens,
            sent_target_lens,
        ) in tqdm.auto.tqdm(datagen_train(), total=epoch_batches_num):
            optim.zero_grad()

            sent_source_batch = sent_source_batch.to(device)
            sent_target_batch = sent_target_batch.to(device)

            sent_target_preds = model(sent_source_batch, sent_target_batch)

            sent_target_preds = sent_target_preds.view(-1, vocab_size_target)
            sent_target_batch = sent_target_batch.reshape(-1)

            loss = criterion(sent_target_preds, sent_target_batch)
            loss.backward()
            optim.step()

            train_acc += calc_acc(
                sent_target_preds, sent_target_batch, tokenizer_fi.pad_id()
            )

            del sent_source_batch
            del sent_target_batch
            del sent_target_preds

        train_acc /= epoch_batches_num

        model.eval()

        print(f"Train acc : {train_acc:.4f}")
        # print(f"Eval acc  : {eval_acc:.4f}")

        if epoch % epochs_per_checkpoint == 0 or epoch == train_epochs:
            print("Saving checkpoint...")
            torch.save(model.state_dict(), checkpoint_path)
            print("Done.")


if __name__ == "__main__":
    _test()
