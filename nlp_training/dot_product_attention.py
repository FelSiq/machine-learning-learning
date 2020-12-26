"""Dot-product attention to perform token alignment.

Often used with Transformers.

There are three common types of attention mechanism:
    - Encoder-Decoder attention (build attention comparing two distinct sequences)
    - Self attention/causal attention (build attention from a single sequence but only previous tokens)
    - Bidirectional self-attention (build attention from a single sequence and all tokens. It is just the
        usual encoder-decoder attention but with sequence(queries) = sequence(keys))
"""
import typing as t

import numpy as np
import scipy.special


def _softmax(x):
    lse_x = scipy.special.logsumexp(x, axis=-1, keepdims=True)
    return np.exp(x - lse_x)


def dp_attention(
    queries, keys, values, mask: t.Optional[np.ndarray] = None, scale: bool = True
):
    embed_dim = queries.shape[-1]
    scale_weight = np.sqrt(embed_dim) if scale else 1.0

    aux = np.dot(queries, np.swapaxes(keys, -1, -2)) / scale_weight

    if mask is not None:
        aux += mask

    attention_weights = _softmax(aux)
    attention_score = np.dot(attention_weights, values)

    return attention_score


def causal_attention(queries, keys, values, scale: bool = True):
    """Also known as 'self attention'.

    A token's attention score can only be calculated from the
    previous tokens of the same sequence.
    """
    mask_shape = (queries.shape[0], keys.shape[0])
    mask = np.triu(np.full(mask_shape, fill_value=-np.inf), k=1)
    attention_score = dp_attention(queries, keys, values, mask, scale=scale)
    return attention_score


def _test():
    q_dim = 5
    v_dim = 7
    d_model = 10

    np.random.seed(16)

    Q = np.random.randn(q_dim, d_model)
    V = np.random.randn(v_dim, d_model)

    att = dp_attention(Q, V, V)
    att = causal_attention(Q, V, V)
    print(att)


if __name__ == "__main__":
    _test()
