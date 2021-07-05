import numpy as np

import modules
import losses
import optimizers


class _BaseTransformerBlock(modules.BaseModel):
    def __init__(
        self,
        dim_embed: int,
        num_embed_tokens: int,
        max_seq_len: int,
    ):
        super(_BaseTransformerBlock, self).__init__()

        self.preprocessing = modules.Sequential(
            [
                modules.Embedding(num_embed_tokens, dim_embed),
                modules.PositionalEncodingLearnable(max_seq_len, dim_embed),
            ]
        )

        self.register_layers(self.preprocessing)


class TransformerEncoder(_BaseTransformerBlock):
    def __init__(
        self,
        dim_embed: int,
        num_embed_tokens: int,
        dim_feedforward: int,
        max_seq_len: int,
        num_attention_heads: int = 8,
        num_blocks: int = 3,
        dropout: float = 0.3,
    ):
        assert int(num_blocks) > 0

        super(TransformerEncoder, self).__init__(
            dim_embed=dim_embed,
            num_embed_tokens=num_embed_tokens,
            max_seq_len=max_seq_len,
        )

        self.blocks = [
            self._create_block(dim_embed, num_attention_heads, dim_feedforward, dropout)
            for _ in range(num_blocks)
        ]

        self.register_layers(*self.blocks)

    def forward(self, X):
        out = self.preprocessing(X)
        num_blocks = len(self.blocks)
        outs = np.zeros((num_blocks, *out.shape), dtype=float)

        for i, block in enumerate(self.blocks):
            out = outs[i, :, :, :] = block(out)

        return outs

    def backward(self, douts):
        dout_a = np.zeros_like(douts[-1], dtype=float)

        for i, block in enumerate(reversed(self.blocks), 1):
            if i <= len(douts):
                dout_b = douts[-i]
                dout_a += dout_b

            dout_a = block.backward(dout_a)

        dout = self.preprocessing.backward(dout_a)

        return dout

    @staticmethod
    def _create_block(
        dim_embed: int, num_attention_heads: int, dim_feedforward: int, dropout: int
    ):
        encoder_block = modules.Sequential(
            [
                modules.SkipConnection(
                    modules.Sequential(
                        [
                            modules.MultiheadAttentionQKV(
                                dim_embed, num_attention_heads
                            ),
                            modules.Dropout(dropout),
                        ]
                    )
                ),
                modules.LayerNorm1d(dim_embed),
                modules.SkipConnection(
                    modules.Sequential(
                        [
                            modules.Linear(dim_in=dim_embed, dim_out=dim_feedforward),
                            modules.GELU(),
                            modules.Linear(dim_in=dim_feedforward, dim_out=dim_embed),
                            modules.Dropout(dropout),
                        ]
                    )
                ),
                modules.LayerNorm1d(dim_embed),
            ]
        )

        return encoder_block


class TransformerDecoder(_BaseTransformerBlock):
    def __init__(
        self,
        dim_embed: int,
        num_embed_tokens: int,
        dim_feedforward: int,
        max_seq_len: int,
        num_attention_heads: int = 8,
        num_blocks: int = 3,
        dropout: float = 0.3,
    ):
        assert int(num_blocks) > 0

        super(TransformerDecoder, self).__init__(
            dim_embed=dim_embed,
            num_embed_tokens=num_embed_tokens,
            max_seq_len=max_seq_len,
        )

        self.blocks_start = []  # type: t.List[base.BaseComponent]
        self.blocks_end = []  # type: t.List[base.BaseComponent]

        for _ in range(num_blocks):
            b_start, b_end = self._create_block(
                dim_embed, num_attention_heads, dim_feedforward, dropout
            )
            self.blocks_start.append(b_start)
            self.blocks_end.append(b_end)
            self.register_layers(b_start, b_end)

    def __call__(self, Y, X_attention_scores):
        return self.forward(Y, X_attention_scores)

    def forward(self, Y, X_attention_scores):
        out = self.preprocessing(Y)

        for b_start, b_end, X_as in zip(
            self.blocks_start, self.blocks_end, X_attention_scores
        ):
            out = block_end(block_start(out), X_as)

        return outs

    def backward(self, dout):
        num_blocks = len(self.blocks_start)
        dX_attention_scores = np.empty((num_blocks, *dout.shape), dtype=float)

        for i, (b_start, b_end, X_as) in reversed(
            list(enumerate(zip(self.blocks_start, self.blocks_end, X_attention_scores)))
        ):
            dout, dout_X_as = block_end.backward(dout)
            dout = block_start.backward(dout)
            dX_attention_scores[i, :, :, :] = dout_X_as

        dout = self.preprocessing.backward(dout)

        return dout

    @staticmethod
    def _create_block(
        dim_embed: int, num_attention_heads: int, dim_feedforward: int, dropout: int
    ):
        decoder_block_start = modules.Sequential(
            [
                modules.SkipConnection(
                    modules.MultiheadMaskedSelfAttentionQKV(
                        dim_embed, num_attention_heads
                    ),
                ),
                modules.LayerNorm1d(dim_embed),
            ]
        )

        decoder_block_end = modules.Sequential(
            [
                modules.SkipConnection(
                    modules.Sequential(
                        [
                            modules.MultiheadAttentionQKV(
                                dim_embed, num_attention_heads
                            ),
                            modules.Dropout(dropout),
                        ]
                    )
                ),
                modules.LayerNorm1d(dim_embed),
                modules.SkipConnection(
                    modules.Sequential(
                        [
                            modules.Linear(dim_in=dim_embed, dim_out=dim_feedforward),
                            modules.GELU(),
                            modules.Linear(dim_in=dim_feedforward, dim_out=dim_embed),
                            modules.Dropout(dropout),
                        ]
                    )
                ),
                modules.LayerNorm1d(dim_embed),
            ]
        )

        return (decoder_block_start, decoder_block_end)


class Transformer(modules.BaseModel):
    def __init__(
        self,
        dim_embed: int,
        num_embed_tokens: int,
        dim_feedforward: int,
        max_seq_len: int,
        dim_out: int,
        num_attention_heads: int = 8,
        num_blocks: int = 3,
        dropout: float = 0.3,
        include_decoder: bool = True,
    ):
        assert int(num_blocks) > 0

        super(Transformer, self).__init__()

        self.encoder = TransformerEncoder(
            dim_embed=dim_embed,
            num_embed_tokens=num_embed_tokens,
            dim_feedforward=dim_feedforward,
            max_seq_len=max_seq_len,
            num_blocks=num_blocks,
            dropout=dropout,
            num_attention_heads=num_attention_heads,
        )

        self.decoder = None

        if include_decoder:
            self.decoder = TransformerDecoder(
                dim_embed=dim_embed,
                num_embed_tokens=num_embed_tokens,
                dim_feedforward=dim_feedforward,
                max_seq_len=max_seq_len,
                num_blocks=num_blocks,
                dropout=dropout,
                num_attention_heads=num_attention_heads,
            )
            self.register_layers(self.decoder)

        self.lin_output = modules.Linear(dim_embed, dim_out)
        self.register_layers(self.encoder, self.lin_output)
        self._cache = []

    def forward(self, X, Y=None):
        out = self.encoder(X)

        if self.decoder is not None:
            out = self.decoder(Y, out)

        out = self.lin_output(out)
        self._cache.append(out.shape)
        return out

    def backward(self, dout):
        # TODO: FIX THIS -----------------
        # (broadcasting bug)
        out_shape = self._cache.pop()
        dout_b = np.zeros(out_shape, dtype=float)
        dout_b[-1, -1, :, :] = dout
        dout = dout_b
        # -------------------------------

        dout = self.lin_output.backward(dout)

        if self.decoder is not None:
            dY, dout = self.decoder.backward(dout)
            dX = self.encoder.backward(dout)
            return dX, dY

        dout = self.encoder.backward(dout)
        return dout


def _test():
    import tqdm.auto
    from test import tweets_utils

    def pad_batch(X):
        lens = np.fromiter(map(len, X), count=len(X), dtype=int)
        batch_max_seq_len = int(max(lens))

        X_padded = np.array(
            [
                np.hstack(
                    (
                        np.array(inst, dtype=int),
                        np.zeros(batch_max_seq_len - len(inst), dtype=int),
                    )
                )
                for inst in X
            ],
            dtype=int,
        )

        return X_padded

    np.random.seed(32)

    batch_size = 64
    train_epochs = 5

    X_train, y_train, X_test, y_test, word_count = tweets_utils.get_data()

    y_train = y_train.ravel().astype(int, copy=False)
    y_test = y_test.ravel().astype(int, copy=False)

    X_eval, X_test = X_test[:50], X_test[50:]
    y_eval, y_test = y_test[:50], y_test[50:]

    token_dictionary = tweets_utils.build_dictionary(word_count, max_token_num=2048)
    tweets_utils.encode_tweets(X_train, token_dictionary)
    tweets_utils.encode_tweets(X_test, token_dictionary)
    tweets_utils.encode_tweets(X_eval, token_dictionary)

    X_test = pad_batch(X_test)
    X_eval = pad_batch(X_eval)

    model = Transformer(
        num_embed_tokens=1 + len(token_dictionary),
        dim_embed=16,
        dim_feedforward=128,
        max_seq_len=64,
        dim_out=1,
        num_blocks=2,
        num_attention_heads=8,
        include_decoder=False,
    )

    criterion = losses.BCELoss(with_logits=True)
    optim = optimizers.Nadam(
        model.parameters,
        learning_rate=1e-3,
        clip_grad_val=0.1,
        demon_min_mom=0.1,
        demon_iter_num=train_epochs * (len(X_train)) / batch_size,
    )

    batch_inds = np.arange(len(X_train))

    for epoch in np.arange(1, 1 + train_epochs):
        total_loss_train = total_loss_eval = 0.0
        it = 0

        np.random.shuffle(batch_inds)
        X_train = [X_train[i] for i in batch_inds]
        y_train = y_train[batch_inds]

        for start in tqdm.auto.tqdm(np.arange(0, len(X_train), batch_size)):
            optim.zero_grad()

            end = start + batch_size
            X_batch = X_train[start:end]
            y_batch = y_train[start:end]

            X_batch = pad_batch(X_batch)

            model.train()
            y_logits = model(X_batch.T)
            y_logits = y_logits[-1][-1]
            loss, loss_grad = criterion(y_batch, y_logits)
            loss_grad = np.expand_dims(loss_grad, (0, 1))
            model.backward(loss_grad)
            total_loss_train += loss

            optim.clip_grads_val()
            optim.step()

            model.eval()
            y_logits = model(X_eval.T)
            y_logits = y_logits[-1][-1]
            loss, _ = criterion(y_eval, y_logits)
            total_loss_eval += loss

            it += 1

        total_loss_train /= it
        total_loss_eval /= it

        print(f"Total loss (train) : {total_loss_train:.3f}")
        print(f"Total loss (eval)  : {total_loss_eval:.3f}")

    model.eval()
    y_preds_logits = model(X_test.T)

    if isinstance(criterion, losses.BCELoss):
        y_preds_logits = np.squeeze(y_preds_logits[-1])
        y_preds = (y_preds_logits > 0.0).astype(int, copy=False)

    else:
        y_preds_logits = y_preds_logits[-1]
        y_preds = y_preds_logits.argmax(axis=-1)

    test_acc = float(np.mean(y_preds == y_test))
    print(f"Test acc: {test_acc:.3f}")


if __name__ == "__main__":
    _test()
