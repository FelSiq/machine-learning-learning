import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster


def _test():
    import sklearn.datasets

    X, _ = sklearn.datasets.load_iris(return_X_y=True)

    n_max = 6
    sse = np.empty(n_max, dtype=float)

    for n_clusters in range(1, n_max + 1):
        model = sklearn.cluster.KMeans(n_clusters=n_clusters)
        model.fit(X)
        sse[n_clusters - 1] = model.inertia_

    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.plot(list(range(1, n_max + 1)), sse)
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("SSE")
    plt.show()


if __name__ == "__main__":
    _test()
