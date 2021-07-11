import typing as t

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

import data.prepare_tensors


class ConvAttentionQKV(nn.Module):
    def __init__(self, dim_in: int, n_heads: int):
        super(ConvAttentionQKV, self).__init__()

        self.attention = nn.MultiheadAttention(
            dim_in,
            num_heads=n_heads,
            batch_first=True,
        )

    def forward(self, conv_maps):
        N, E, H, W = conv_maps.size()
        conv_maps = conv_maps.view(N, E, H * W)
        conv_maps = conv_maps.swapaxes(1, 2)
        out, _ = self.attention(conv_maps, conv_maps, conv_maps)
        out = out.swapaxes(1, 2)
        out = out.view(N, E, H, W)
        return out


class ConvAttentionChannel(nn.Module):
    def __init__(self, dim_in: int, bottleneck_ratio: float = 0.20):
        super(ConvAttentionChannel, self).__init__()

        bottleneck_size = int(np.ceil(bottleneck_ratio * dim_in))

        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dim_in, bottleneck_size),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_size, dim_in),
        )

    def forward(self, X):
        spatial_shape = X.size()[2:]
        chan_num = X.size()[1]

        C_p_max = F.max_pool2d(X, kernel_size=spatial_shape)
        C_p_avg = F.avg_pool2d(X, kernel_size=spatial_shape)

        mlp_max = self.mlp(C_p_max)
        mlp_avg = self.mlp(C_p_avg)

        mlp_logits = mlp_max + mlp_avg

        weights = torch.sigmoid(mlp_logits)
        weights = weights.view(-1, chan_num, 1, 1)

        out = X * weights

        return out


class ConvAttentionSpatial(nn.Module):
    def __init__(self, dim_in: int):
        super(ConvAttentionSpatial, self).__init__()
        self.conv_combine_pools = nn.Conv2d(2, 1, kernel_size=3, padding=1)

    def forward(self, X):
        spatial_shape = X.size()[2:]
        spatial_size = int(np.prod(spatial_shape))
        chan_num = X.size()[1]

        C_p_max, _ = X.max(axis=1, keepdim=True)
        C_p_avg = X.mean(axis=1, keepdim=True)

        C = torch.cat((C_p_max, C_p_avg), axis=1)

        weights = self.conv_combine_pools(C)
        weights = torch.sigmoid(weights)

        out = X * weights

        return out


class ConvFullAttentionBlock(nn.Module):
    def __init__(self, dim_in: int, n_heads: int):
        super(ConvFullAttentionBlock, self).__init__()

        self.conv_att_chan = ConvAttentionChannel(dim_in)
        self.conv_att_spatial = ConvAttentionSpatial(dim_in)
        # self.conv_att_qkv = ConvAttentionQKV(dim_in, n_heads)
        # self.conv_final = nn.Conv2d(2 * dim_in, dim_in, 1, bias=False)

    def forward(self, X):
        out_a = out_b = X

        out_a = self.conv_att_chan(out_a)
        out_a = self.conv_att_spatial(out_a)

        # out_b = self.conv_att_qkv(out_b)

        # out = torch.cat((out_a, out_b), axis=1)
        # out = self.conv_final(out)
        out = out_a

        return out


class AttentionCNN(nn.Module):
    @staticmethod
    def _compute_out_dim(dim_in, pool=True):
        dim_out = 1 + (dim_in - 3)

        if pool:
            dim_out //= 2

        return dim_out

    def __init__(self, image_shape, dims, n_heads, dim_emb, dropout: float):
        super(AttentionCNN, self).__init__()

        dim_dense_height, dim_dense_width = image_shape

        for i in range(len(dims) - 1):
            dim_dense_height = self._compute_out_dim(dim_dense_height, pool=True)
            dim_dense_width = self._compute_out_dim(dim_dense_width, pool=True)

        dim_dense = dim_dense_height * dim_dense_width * dims[-1]
        dim_hidden = (dim_dense + dim_emb) // 2

        self.weights = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(dims[i - 1], dims[i], 3),
                    ConvFullAttentionBlock(dims[i], n_heads),
                    nn.BatchNorm2d(dims[i]),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Dropout2d(dropout, inplace=False),
                )
                for i in range(1, len(dims))
            ],
            nn.Flatten(),
            nn.Linear(dim_dense, dim_hidden, bias=False),
            nn.BatchNorm1d(dim_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim_emb),
        )

    def forward(self, X):
        return self.weights(X)


class PositionalEncoding(nn.Module):
    # Source: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, dim_emb, dropout=0.0, max_len=40):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, dim_emb)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim_emb, 2).float() * (-float(np.log(10000.0)) / dim_emb)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class CaptionGenerator(nn.Module):
    def __init__(
        self,
        image_shape: t.Tuple[int, int],
        dims: t.Tuple[int, ...],
        dim_emb: int,
        num_tokens: int,
        n_heads_cnn: int,
        n_heads_transf: int,
        num_layers: int,
        dropout: float,
    ):
        super(CaptionGenerator, self).__init__()
        self.att_cnn = AttentionCNN(image_shape, dims, n_heads_cnn, dim_emb, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_emb, nhead=n_heads_transf
        )

        self.embed = nn.Sequential(
            nn.Embedding(num_tokens, dim_emb),
            PositionalEncoding(dim_emb, dropout),
        )

        self.transf_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.final_lin = nn.Linear(dim_emb, num_tokens)

    def _build_mask(self, description):
        desc_lenp1 = 1 + len(description)
        mask = torch.triu(
            torch.full(
                (desc_lenp1, desc_lenp1), fill_value=True, device=description.device
            ),
            diagonal=1,
        )
        return mask

    def forward(self, img, description, img_embed=None):
        if img_embed is None:
            img_embed = self.att_cnn(img)
            img_embed = torch.unsqueeze(img_embed, 0)

        desc_embed = self.embed(description)

        transf_in = torch.cat((img_embed, desc_embed), axis=0)
        mask = self._build_mask(desc_embed)
        out = self.transf_encoder(transf_in, mask=mask)
        out = self.final_lin(out)

        # NOTE: removing output related to the image embedding
        out = out[1:, ...]

        return out, img_embed


def logsoftmax_sample(logits, temperature=1.0):
    assert 0 <= temperature <= 1.0

    log_probs = nn.functional.log_softmax(logits, dim=-1)
    # Note: sample uniform U(1e-6, 1 - 1e-6)
    u = torch.rand_like(log_probs) * (1 - 2e-6) + 1e-6
    g = -torch.log(-torch.log(u))
    return torch.argmax(log_probs + g * temperature, axis=-1)


def generate(
    model,
    img,
    max_length: int,
    vocab,
    vocab_inv,
    device,
    temperature: float = 1.0,
):
    model.eval()

    output = torch.zeros(max_length, dtype=torch.long)
    output[0] = vocab["<SOS>"]

    output = output.unsqueeze(1)
    output = output.to(device)

    img = img.unsqueeze(0)
    img = img.to(device)

    img_embed = None

    for i in range(1, max_length):
        cur_output, img_embed = model(img, output, img_embed=img_embed)
        logits = cur_output[i]
        next_token = logsoftmax_sample(logits, temperature)

        if next_token == vocab["<EOS>"]:
            break

        output[i] = next_token.item()

    out = [vocab_inv[int(token.item())] for token in output]
    result = " ".join(out)

    return result


def _test():
    import tqdm.auto

    np.random.seed(16)

    def pad_desc(y, pad_id: int = 0):
        y_lens = list(map(len, y))
        y_padded = nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=pad_id)
        return y_padded, y_lens

    device = "cuda"
    train_epochs = 100
    lr = 1e-3

    (
        dataloader_train,
        dataloader_eval,
        vocab,
        vocab_inv,
    ) = data.prepare_tensors.get_data(batch_size_train=32)

    model = CaptionGenerator(
        dims=[3, 64, 128, 64],
        image_shape=(32, 32),
        dim_emb=32,
        num_tokens=len(vocab),
        n_heads_cnn=4,
        n_heads_transf=8,
        num_layers=3,
        dropout=0.2,
    )

    modl = model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, factor=0.1, patience=5, mode="min"
    )
    criterion = nn.CrossEntropyLoss(ignore_index=vocab["<PAD>"])

    for epoch in np.arange(1, 1 + train_epochs):
        total_loss_train = 0.0
        it_train = 0

        total_loss_eval = 0.0
        it_eval = 0

        for X_batch, y_batch in tqdm.auto.tqdm(dataloader_train):
            y_batch, _ = pad_desc(y_batch)
            y_batch = y_batch.transpose(0, 1)

            y_batch = y_batch.to(device)
            X_batch = X_batch.to(device)

            y_batch_inp = y_batch[:-1, ...]
            y_batch_target = y_batch[1:, ...]

            optim.zero_grad()
            y_preds, _ = model(X_batch, y_batch_inp)
            y_preds = y_preds.view(-1, len(vocab))
            y_batch_target = y_batch_target.reshape(-1)

            loss = criterion(y_preds, y_batch_target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            it_train += 1
            total_loss_train += loss.item()

        for X_batch, y_batch in tqdm.auto.tqdm(dataloader_eval):
            y_batch, _ = pad_desc(y_batch)
            y_batch = y_batch.transpose(0, 1)

            y_batch = y_batch.to(device)
            X_batch = X_batch.to(device)

            y_batch_inp = y_batch[:-1, ...]
            y_batch_target = y_batch[1:, ...]

            y_preds, _ = model(X_batch, y_batch_inp)
            y_preds = y_preds.view(-1, len(vocab))
            y_batch_target = y_batch_target.reshape(-1)

            loss = criterion(y_preds, y_batch_target)

            it_eval += 1
            total_loss_eval += loss.item()

        total_loss_train /= it_train
        total_loss_eval /= it_eval

        scheduler.step(total_loss_eval)

        print(f"Loss train: {total_loss_train:.3f}")
        print(f"Loss eval: {total_loss_eval:.3f}")

    img_train = next(iter(dataloader_train))[0][0]
    img_eval = next(iter(dataloader_eval))[0][0]

    result_train = generate(
        model,
        img_train,
        20,
        vocab,
        vocab_inv,
        device,
        temperature=0.15,
    )

    result_eval = generate(
        model,
        img_eval,
        20,
        vocab,
        vocab_inv,
        device,
        temperature=0.15,
    )

    fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 10))

    ax1.imshow(np.moveaxis(img_train.numpy(), 0, -1).astype(int))
    ax1.set_title(result_train)

    ax2.imshow(np.moveaxis(img_eval.numpy(), 0, -1).astype(int))
    ax2.set_title(result_eval)

    plt.show()


if __name__ == "__main__":
    _test()
