import typing as t

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import bpemb

import data.prepare_tensors


class PositionalEncoding(nn.Module):
    def __init__(self, dim_emb, dropout=0.0, max_len=80):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pos_enc = nn.Parameter(
            torch.zeros((max_len, 1, dim_emb), dtype=torch.float), requires_grad=True
        )

    def forward(self, X):
        seq_len = X.size(0)
        X = X + self.pos_enc[:seq_len, :]
        return self.dropout(X)


class CaptionGenerator(nn.Module):
    def __init__(
        self,
        codec,
        dim_emb: int,
        n_heads_transf: int,
        num_layers: int,
        dropout: float,
    ):
        super(CaptionGenerator, self).__init__()
        num_tokens = codec.vocab_size
        codec_dim = codec.dim
        self.pad_id = num_tokens

        cnn = torchvision.models.resnet18(pretrained=True).eval()

        for param in cnn.parameters():
            param.requires_grad = False

        self.img_embed = nn.Sequential(
            cnn.conv1,
            cnn.bn1,
            cnn.relu,
            cnn.maxpool,
            cnn.layer1,
            cnn.layer2,
            cnn.layer3,
            cnn.layer4,
            cnn.avgpool,
            nn.Flatten(),
            nn.Linear(512, dim_emb, bias=False),
            nn.BatchNorm1d(dim_emb),
        )

        emb_tensor = torch.from_numpy(codec.vectors.astype(np.float32, copy=False))

        self.desc_embed = nn.Sequential(
            nn.Embedding.from_pretrained(
                emb_tensor, padding_idx=self.pad_id, freeze=False
            ),
            PositionalEncoding(codec_dim, dropout),
            nn.Linear(codec_dim, dim_emb),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_emb,
            nhead=n_heads_transf,
            activation="gelu",
            dim_feedforward=512,
            dropout=0.5 * dropout,
        )

        self.transf_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.final_lin = nn.Linear(dim_emb, num_tokens)

    def _build_mask_attention(self, description):
        desc_lenp1 = 1 + len(description)
        mask = torch.triu(
            torch.full(
                (desc_lenp1, desc_lenp1),
                fill_value=True,
                device=description.device,
            ),
            diagonal=1,
        )
        return mask

    def _build_mask_padding(self, description):
        batch_size = description.size(1)
        img_embed_row = torch.full((1, batch_size), False).to(description.device)
        mask = torch.cat((img_embed_row, description == self.pad_id), dim=0)
        return mask.transpose(0, 1)

    def forward(self, img, description, img_embed=None):
        if img_embed is None:
            img_embed = self.img_embed(img)
            img_embed = torch.unsqueeze(img_embed, 0)

        desc_embed = self.desc_embed(description)

        transf_in = torch.cat((img_embed, desc_embed), axis=0)

        mask = pad_mask = None

        if self.training:
            mask = self._build_mask_attention(description)
            pad_mask = self._build_mask_padding(description)

        out = self.transf_encoder(transf_in, mask, pad_mask)

        # NOTE: removing output related to the image embedding
        out = out[1:, ...]

        out = self.final_lin(out)

        return out, img_embed


def logsoftmax_sample(logits, temperature=0.0):
    assert 0 <= temperature <= 1.0
    log_probs = nn.functional.log_softmax(logits, dim=-1)
    u = torch.rand_like(log_probs) * (1 - 2e-6) + 1e-6
    g = -torch.log(-torch.log(u))
    return torch.argmax(log_probs + g * temperature, axis=-1)


def generate(
    model,
    img,
    max_length: int,
    codec,
    device,
    start_of_sentence=None,
):
    model.eval()

    if start_of_sentence is None:
        output = torch.tensor([[codec.BOS]], dtype=torch.long).to(device)

    else:
        output = start_of_sentence.unsqueeze(1).to(device)

    img = img.unsqueeze(0)
    img = img.to(device)

    img_embed = None

    for i in range(max_length - len(output)):
        cur_output, img_embed = model(img, output, img_embed=img_embed)
        logits = cur_output[-1].unsqueeze(0)
        next_token = logsoftmax_sample(logits)
        output = torch.cat((output, next_token), dim=0)

        if next_token.item() == codec.EOS:
            break

    raw = output.detach().cpu().squeeze().tolist()
    print("raw:", raw)
    result = codec.decode_ids(raw)

    return result


def _test():
    import tqdm.auto

    np.random.seed(16)
    torch.random.manual_seed(32)

    def pad_desc(y, pad_id: int):
        y_lens = list(map(len, y))
        y_padded = nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=pad_id)
        return y_padded, y_lens

    device = "cuda"
    train_epochs = 5
    lr = 3e-4
    img_shape = (224, 224)

    (dataloader_train, dataloader_eval, codec,) = data.prepare_tensors.get_data(
        batch_size_train=64, img_shape=img_shape, vs=3000, dim=100
    )

    model = CaptionGenerator(
        codec=codec,
        dim_emb=256,
        n_heads_transf=8,
        num_layers=6,
        dropout=0.4,
    )

    model = model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, factor=0.1, patience=5, mode="min"
    )
    criterion = nn.CrossEntropyLoss(ignore_index=codec.vocab_size)

    for epoch in np.arange(1, 1 + train_epochs):
        total_loss_train = 0.0
        it_train = 0

        total_loss_eval = 0.0
        it_eval = 0

        model.train()

        for X_batch, y_batch in tqdm.auto.tqdm(dataloader_train):
            y_batch, _ = pad_desc(y_batch, pad_id=codec.vocab_size)
            y_batch = y_batch.transpose(0, 1)

            y_batch = y_batch.to(device)
            X_batch = X_batch.to(device)

            y_batch_inp = y_batch[:-1, ...]
            y_batch_target = y_batch[1:, ...]

            optim.zero_grad()
            y_preds, _ = model(X_batch, y_batch_inp)
            y_preds = y_preds.view(-1, codec.vocab_size)
            y_batch_target = y_batch_target.reshape(-1)

            loss = criterion(y_preds, y_batch_target)
            loss.backward()

            try:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            except RuntimeError:
                nn.utils.clip_grad_value_(model.parameters(), 1.0)

            optim.step()

            it_train += 1
            total_loss_train += loss.item()

        model.eval()

        for X_batch, y_batch in tqdm.auto.tqdm(dataloader_eval):
            y_batch, _ = pad_desc(y_batch, pad_id=codec.vocab_size)
            y_batch = y_batch.transpose(0, 1)

            y_batch = y_batch.to(device)
            X_batch = X_batch.to(device)

            y_batch_inp = y_batch[:-1, ...]
            y_batch_target = y_batch[1:, ...]

            y_preds, _ = model(X_batch, y_batch_inp)
            y_preds = y_preds.view(-1, codec.vocab_size)
            y_batch_target = y_batch_target.reshape(-1)

            loss = criterion(y_preds, y_batch_target)

            it_eval += 1
            total_loss_eval += loss.item()

        total_loss_train /= it_train
        total_loss_eval /= it_eval

        scheduler.step(total_loss_eval)

        print(f"Loss train: {total_loss_train:.3f}")
        print(f"Loss eval: {total_loss_eval:.3f}")

    X_test_batch_train, y_test_batch_train = next(iter(dataloader_train))
    X_test_batch_eval, y_test_batch_eval = next(iter(dataloader_eval))

    img_train = X_test_batch_train[0]
    img_eval = X_test_batch_eval[0]

    exp_caption_train = y_test_batch_train[0]
    exp_caption_eval = y_test_batch_eval[0]

    result_train = generate(
        model,
        img_train,
        30,
        codec,
        device,
    )

    result_eval = generate(
        model,
        img_eval,
        30,
        codec,
        device,
    )

    fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 10))

    ax1.imshow(np.moveaxis(img_train.numpy(), 0, -1))
    ax1.set_title(result_train + f"\nExpected: {codec.decode_ids(exp_caption_train)}")

    ax2.imshow(np.moveaxis(img_eval.numpy(), 0, -1))
    ax2.set_title(result_eval + f"\nExpected: {codec.decode_ids(exp_caption_eval)}")

    plt.show()


if __name__ == "__main__":
    _test()
