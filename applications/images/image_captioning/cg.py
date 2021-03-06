import typing as t

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import bpemb

import data.prepare_tensors


class SkipConnection(nn.Module):
    def __init__(self, layer):
        super(SkipConnection, self).__init__()
        self.layer = layer

    def forward(self, X):
        return self.layer(X) + X



class ConvAttentionQKV(nn.Module):
    # NOTE: consumes too much GPU memory and processing time for my taste.

    def __init__(self, dim_in: int, n_heads: int):
        super(ConvAttentionQKV, self).__init__()

        self.attention = nn.MultiheadAttention(
            dim_in,
            num_heads=n_heads,
            batch_first=True,
        )

    def forward(self, conv_maps):
        N, E, H, W = conv_maps.size()

        if H * W == 1:
            return conv_maps

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
    def __init__(self, dim_in: int):
        super(ConvFullAttentionBlock, self).__init__()

        self.conv_att_chan = ConvAttentionChannel(dim_in)
        self.conv_att_spatial = ConvAttentionSpatial(dim_in)

    def forward(self, X):
        out = X
        out = self.conv_att_chan(out)
        out = self.conv_att_spatial(out)
        return out


class AttentionCNN(nn.Module):
    KERNEL_SIZE = 3

    @classmethod
    def _compute_out_dim(cls, dim_in, pool=True, padding=False):
        if padding:
            dim_out = dim_in

        else:
            dim_out = 1 + (dim_in - cls.KERNEL_SIZE)

        if pool:
            dim_out //= 2

        return dim_out

    def __init__(
        self,
        image_shape,
        dims,
        dim_emb,
        n_heads: int,
        dropout: float,
        pool: bool = True,
        padding: bool = True,
    ):
        super(AttentionCNN, self).__init__()

        dim_dense_height, dim_dense_width = image_shape

        for i in range(len(dims) - 1):
            dim_dense_height = self._compute_out_dim(
                dim_dense_height, pool=pool, padding=padding
            )
            dim_dense_width = self._compute_out_dim(
                dim_dense_width, pool=pool, padding=padding
            )

        dim_dense = dim_dense_height * dim_dense_width * dims[-1]
        dim_hidden = (dim_dense + dim_emb) // 2
        padding = self.KERNEL_SIZE // 2 if padding else 0

        self.weights = nn.Sequential(
            *[
                nn.Sequential(
                    self._create_qkv_att_block(dims[i - 1], n_heads),
                    nn.Conv2d(dims[i - 1], dims[i], self.KERNEL_SIZE, padding=padding),
                    ConvFullAttentionBlock(dims[i]),
                    nn.BatchNorm2d(dims[i]),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2) if pool else nn.Identity(),
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

    @staticmethod
    def _create_qkv_att_block(dim_in, n_heads):
        if dim_in % n_heads != 0:
            return nn.Identity()

        block = nn.Sequential(
            SkipConnection(ConvAttentionQKV(dim_in, n_heads)),
            nn.BatchNorm2d(dim_in),
        )

        return block

    def forward(self, X):
        return self.weights(X)


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
        image_shape: t.Tuple[int, int],
        codec,
        dim_emb: int,
        dims: t.Tuple[int, ...],
        n_heads_conv: int,
        n_heads_transf: int,
        num_layers_transf: int,
        dropout: float,
    ):
        super(CaptionGenerator, self).__init__()

        num_tokens = codec.vocab_size
        self.pad_id = num_tokens
        codec_dim = codec.dim

        self.att_cnn = AttentionCNN(
            image_shape=image_shape,
            dims=dims,
            dim_emb=dim_emb,
            n_heads=n_heads_conv,
            dropout=dropout,
        )
        emb_tensor = torch.from_numpy(codec.vectors.astype(np.float32, copy=False))

        self.embed_desc = nn.Sequential(
            nn.Embedding.from_pretrained(
                emb_tensor, padding_idx=self.pad_id, freeze=False
            ),
            PositionalEncoding(codec_dim, dropout),
            nn.Linear(codec_dim, dim_emb, bias=True),
        )

        self.layer_norm = nn.LayerNorm(dim_emb)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_emb,
            nhead=n_heads_transf,
            activation="gelu",
            dim_feedforward=1024,
            dropout=0.5 * dropout,
        )

        self.transf_encoder = nn.TransformerEncoder(encoder_layer, num_layers_transf)
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
            img_embed = self.att_cnn(img)
            img_embed = torch.unsqueeze(img_embed, 0)

        desc_embed = self.embed_desc(description)

        transf_in = torch.cat((img_embed, desc_embed), axis=0)
        transf_in = self.layer_norm(transf_in)

        mask_attention = self._build_mask_attention(description)
        mask_pad = self._build_mask_padding(description)

        out = self.transf_encoder(transf_in, mask_attention, mask_pad)
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
        temperature = np.exp(-len(cur_output))
        next_token = logsoftmax_sample(logits, temperature)
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
    train_epochs = 15
    lr = 2e-4
    img_shape = (32, 32)
    checkpoint_uri = "checkpoint.tar"

    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomHorizontalFlip(0.5),
            torchvision.transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
            torchvision.transforms.RandomAffine(degrees=15, scale=(0.9, 1.1)),
        ]
    )

    (dataloader_train, dataloader_eval, codec,) = data.prepare_tensors.get_data(
        batch_size_train=32,
        img_shape=img_shape,
        vs=3000,
        dim=100,
        transforms=transforms,
    )

    model = CaptionGenerator(
        dims=[3, 256, 256, 128, 64],
        codec=codec,
        dim_emb=256,
        image_shape=img_shape,
        n_heads_conv=8,
        n_heads_transf=8,
        num_layers_transf=8,
        dropout=0.3,
    )

    modl = model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, factor=0.1, patience=5, mode="min"
    )

    try:
        checkpoint = torch.load(checkpoint_uri)
        model = model.to(device)
        model.load_state_dict(checkpoint["model"])
        optim.load_state_dict(checkpoint["optim"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        print("Loaded checkpoint successfully.")

    except FileNotFoundError:
        pass

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

        print(f"Epoch: {epoch} / {train_epochs}")
        print(f"Loss train: {total_loss_train:.3f}")
        print(f"Loss eval: {total_loss_eval:.3f}")

    checkpoint = {
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "scheduler": scheduler.state_dict(),
    }

    torch.save(checkpoint, checkpoint_uri)

    img_train = next(iter(dataloader_train))[0][0]
    img_eval = next(iter(dataloader_eval))[0][0]

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
    ax1.set_title(result_train)

    ax2.imshow(np.moveaxis(img_eval.numpy(), 0, -1))
    ax2.set_title(result_eval)

    plt.show()


if __name__ == "__main__":
    _test()
