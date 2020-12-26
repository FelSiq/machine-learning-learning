"""Dot-product attention to perform token alignment.

Often used with Transformers.

There are three common types of attention mechanism:
    - Encoder-Decoder attention (build attention comparing two distinct sequences)
    - Self attention/causal attention (build attention from a single sequence but only previous tokens)
    - Bidirectional attention (build attention from a single sequence and all tokens)
"""
import numpy as np


def _softmax(x):
    x_exp = np.exp(x - np.max(x))
    return x_exp / np.sum(x_exp)


def dp_attention(queries, keys, values):
    attention_weights = _softmax(np.dot(queries, keys.T))
    attention_score = np.dot(attention_weights, values)
    return attention_score


def causal_attention(queries, keys, values):
    """Also known as 'self attention'.

    A token's attention score can only be calculated from the
    previous tokens of the same sequence.
    """
    mask_shape = (queries.shape[0], keys.shape[0])
    mask = np.triu(np.full(mask_shape, fill_value=-np.inf), k=1)
    attention_weights = _softmax(np.dot(queries, keys.T) + mask)
    attention_score = np.dot(attention_weights, values)
    return attention_score


def _test():
    q_dim = 5
    v_dim = 7
    d_model = 10

    Q = np.random.randn(q_dim, d_model)
    V = np.random.randn(v_dim, d_model)

    att = dp_attention(Q, V, V)
    att = causal_attention(Q, V, V)
    print(att)


if __name__ == "__main__":
    _test()
