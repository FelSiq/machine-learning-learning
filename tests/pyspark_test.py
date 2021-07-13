"""Kmeans implementation using pyspark."""
import pyspark
import pyspark.ml
import numpy as np
import matplotlib.pyplot as plt


def closest_point(coords, centroids):
    return np.linalg.norm(centroids - coords, ord=2, axis=-1).argmin()


def kmeans(raw_data, k: int = 3, threshold: float = 1e-3):
    cur_dist = 2 * threshold + 1.0

    data = raw_data.rdd.map(lambda v: np.array(v, dtype=np.float32)).cache()
    centroids = data.takeSample(withReplacement=False, num=k, seed=16)

    while cur_dist > threshold:
        cur_dist = 0

        closest = data.map(lambda p: (closest_point(p, centroids), (p, 1)))

        cluster_stats = closest.reduceByKey(
            lambda p1_f1, p2_f2: (p1_f1[0] + p2_f2[0], p1_f1[1] + p2_f2[1])
        )

        new_centroids = cluster_stats.map(
            lambda idk: (idk[0], idk[1][0] / idk[1][1])
        ).collect()  # Note: the code is actually executed only here

        cur_dist = np.linalg.norm(
            [centroids[iK] - new_centroid for (iK, new_centroid) in new_centroids],
            ord=2,
        ).sum()

        for (iK, new_centroid) in new_centroids:
            centroids[iK] = new_centroid

    centroid_ids = closest.map(lambda p: p[0]).collect()

    return centroids, centroid_ids


def get_pca(spark, raw_data, centroids, k: int = 2):
    assembler = pyspark.ml.feature.VectorAssembler(
        inputCols=raw_data.columns, outputCol="assembled_features"
    )
    vectors = assembler.transform(raw_data).select("assembled_features")

    model = pyspark.ml.feature.PCA(
        k=k, inputCol="assembled_features", outputCol="pca_features"
    ).fit(vectors)

    result = model.transform(vectors).select("pca_features")

    results = np.array(result.collect(), dtype=np.float32).squeeze()

    centroids = list(map(lambda c: (pyspark.ml.linalg.Vectors.dense(c),), centroids))
    centroids = spark.createDataFrame(centroids, schema=["assembled_features"])
    centroids = model.transform(centroids).select("pca_features")
    centroids = np.vstack(centroids.collect())

    return centroids, results


def _test():
    k_kmeans = 3
    k_pca = 2

    spark = (
        pyspark.sql.SparkSession.builder.appName("k-means-test")
        .config("spark.sql.shuffle.partitions", 10)
        .getOrCreate()
    )

    print(spark.conf.get("spark.sql.autoBroadcastJoinThreshold"))

    raw_data = (
        spark.read.option("header", True).option("inferSchema", True).csv("iris.csv")
    )

    centroids, centroid_ids = kmeans(raw_data, k=k_kmeans)
    centroids, pc_features = get_pca(spark, raw_data, centroids, k=k_pca)

    colors = dict(zip(range(3), ["r", "g", "b"]))
    plt.scatter(*pc_features.T, c=list(map(colors.get, centroid_ids)))
    plt.scatter(*centroids.T, marker="X", s=121, c="black", label="centroids")
    plt.legend()
    plt.show()

    spark.stop()


if __name__ == "__main__":
    _test()
