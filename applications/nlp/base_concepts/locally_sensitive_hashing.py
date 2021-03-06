import typing as t
import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial
import collections
import sklearn.preprocessing

import tweets_utils


def cosine_similarity(X_query: np.ndarray, X_subset: np.ndarray) -> np.ndarray:
    return np.dot(X_query, X_subset.T) / (
        1e-7 + np.linalg.norm(X_query) * np.linalg.norm(X_subset, axis=1)
    )


def knn(
    X_query: np.ndarray, X_subset: np.ndarray, k: int = 1, cosine: bool = True
) -> np.ndarray:
    if cosine:
        vals = -cosine_similarity(X_query, X_subset)

    else:
        vals = np.linalg.norm(X_subset - X_query)

    nn = np.argsort(vals)

    return nn[:k]


def create_random_planes(n_planes: int, planes_dim: int) -> np.ndarray:
    return np.random.randn(n_planes, planes_dim)


def hash_instances(X_embed: np.ndarray, planes: np.ndarray) -> np.ndarray:
    n_planes = len(planes)
    in_reg = (np.dot(X_embed, planes.T) >= 0).astype(int, copy=False)
    base = np.geomspace(1, 2 ** (n_planes - 1), n_planes).astype(int)
    hash_vals = np.dot(base, in_reg.T)
    return hash_vals.squeeze()


def embed_document(
    proc_document: t.Sequence[str],
    embeddings: t.Dict[str, np.ndarray],
    embed_dim: int,
) -> np.ndarray:
    doc_embed = np.zeros(embed_dim)

    for w in proc_document:
        doc_embed += embeddings.get(w, 0)

    return doc_embed


def lsh_train(
    X_train,
    embeddings: t.Dict[str, np.ndarray],
    n_universes: int,
    embed_dim: int,
    n_planes: int = 8,
) -> t.Dict[str, t.Any]:
    assert n_universes > 0

    X_train_embed = np.vstack(
        [embed_document(doc, embeddings, embed_dim) for doc in X_train]
    )

    planes = []
    hashes = []

    for _ in np.arange(n_universes):
        cur_planes = create_random_planes(n_planes, embed_dim)
        cur_hash_table = collections.defaultdict(list)

        hash_vals = hash_instances(X_train_embed, cur_planes)

        for ind_inst, ind_hash in enumerate(hash_vals):
            cur_hash_table[ind_hash].append(ind_inst)

        hashes.append(cur_hash_table)
        planes.append(cur_planes)

    return {
        "hashes": hashes,
        "planes": planes,
        "embed_dim": embed_dim,
        "X_train_embed": X_train_embed,
    }


def lsh_predict(
    X_test: np.ndarray,
    embeddings: t.Dict[str, np.ndarray],
    model: t.Dict[str, t.Any],
    k: int = 1,
) -> np.ndarray:
    hashes = model["hashes"]
    planes = model["planes"]
    embed_dim = model["embed_dim"]
    X_train_embed = model["X_train_embed"]

    n_universes = len(planes)

    X_test_embed = np.vstack(
        [embed_document(doc, embeddings, embed_dim) for doc in X_test]
    )

    votes = [collections.Counter() for _ in np.arange(len(X_test))]

    for u in np.arange(n_universes):
        cur_planes = planes[u]
        cur_hash_table = hashes[u]
        hash_vals = hash_instances(X_test_embed, cur_planes)

        for i, inst_embed in enumerate(X_test_embed):
            closest = knn(
                inst_embed, X_train_embed[cur_hash_table[hash_vals[i]], :], k=k
            )
            train_inds = [cur_hash_table[hash_vals[i]][v] for v in closest]
            votes[i].update(train_inds)

    preds = np.zeros(len(votes), dtype=np.uint)

    for i, cur_votes in enumerate(votes):
        chosen = cur_votes.most_common(1)
        if chosen:
            preds[i], _ = chosen[0]

    return preds


def get_embeddings(X_train) -> t.Dict[str, np.ndarray]:
    with open("embedding_models/embedding_model_2_512.pickle", "rb") as f:
        model = pickle.load(f)

    embeddings = model["embeddings"]

    embeddings = sklearn.preprocessing.StandardScaler().fit_transform(embeddings)

    return dict(zip(model["sorted_vocab"], embeddings))


def _test(n_universes: int = 16, k: int = 1):
    X_train, _, X_test, _, freq_pos, freq_neg = tweets_utils.get_data(train_size=4980)
    embeddings = get_embeddings(X_train)
    embed_dim = len(next(iter(embeddings.values())))
    print("Embed dim:", embed_dim)
    model = lsh_train(X_train, embeddings, n_universes, embed_dim, n_planes=8)
    preds = lsh_predict(X_test, embeddings, model, k=3)

    for i, inst in enumerate(X_test):
        print(i, "test:", " ".join(inst), "closest:", " ".join(X_train[preds[i]]))


if __name__ == "__main__":
    _test()
