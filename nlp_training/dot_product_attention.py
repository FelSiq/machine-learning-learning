"""Dot-product attention to perform token alignment.

Often used with Transformers.
"""
import numpy as np


def _softmax(x):
    x_exp = np.exp(x - np.max(x))
    return x_exp / np.sum(x_exp)


def dp_attention(queries, keys, values):
    attention_weights = _softmax(np.dot(queries, keys.T))
    attention_score = np.dot(attention_weights, values)
    return attention_score


def _test():
    q_dim = 5
    v_dim = 7
    d_model = 10

    Q = np.random.randn(q_dim, d_model)
    V = np.random.randn(v_dim, d_model)

    att = dp_attention(Q, V, V)


if __name__ == "__main__":
    _test()
