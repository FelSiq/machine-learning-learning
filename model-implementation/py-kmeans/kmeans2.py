# Helpful video: (in Portuguese)
# title: "Vídeo 48 - Planejamento de Experimentos: Agrupamento de Dados"
# canal: ML4U
# https://www.youtube.com/watch?v=OLwabj1WJj0

from numpy import array, random, argmin, mean as npmean, inf
from math import isnan

import sys
sys.path.insert(0, "../../validation-framework/")
from clustering import ClusterMetrics


class Kmeans():
    def __euclideandist__(inst_a, inst_b):
        return (sum((inst_a - inst_b)**2.0))**0.5

    def run(dataset,
            k,
            it_max=1000,
            min_variation=1.0e-4,
            labels=None,
            full_output=True):

        # Init centers_coord
        centers_id = random.randint(dataset.shape[0], size=k)
        centers_coord = dataset[centers_id, :]

        # Each instance will be put on a initial random cluster
        inst_cluster_id = random.randint(k, size=dataset.shape[0])

        # Auxiliary vectors to keep code cleaner
        auxvec_cur_distances = array([0.0] * k)

        prev_variation = 1.0 + min_variation
        it = 0
        while it < it_max and prev_variation >= min_variation:
            it += 1

            for inst_id in range(dataset.shape[0]):
                nearest_center = inst_cluster_id[inst_id]
                for center_id in range(k):
                    # For each instance, calculate the distance between
                    # each center
                    auxvec_cur_distances[center_id] = \
                     Kmeans.__euclideandist__(\
                      dataset[inst_id,:],
                      centers_coord[center_id,:])

                # For each instance, let it be part of the nearest
                # cluster
                inst_cluster_id[inst_id] = argmin(auxvec_cur_distances)

            # For each cluster, calculate the new center coordinates
            for center_id in range(k):

                part_aux = dataset[inst_cluster_id == center_id, :]

                if part_aux.shape[0] > 0:
                    new_cur_cluster_coords = npmean(part_aux, axis=0)

                    # Calculate variation between previous centers_coord and
                    # new ones (using infinite norm)
                    prev_variation = max(prev_variation, \
                     max(abs(centers_coord[center_id] - \
                      new_cur_cluster_coords)))

                    centers_coord[center_id] = new_cur_cluster_coords

        # Build up answer
        ans = {
            "inst_cluster_id": inst_cluster_id,
            "centers_coord": centers_coord,
        }

        if full_output:
            ans = {
             "clustering_method" : "K-Means",
             "k" : k,
             **ans,
             **ClusterMetrics.runall(
              dataset=dataset,
              centers_coord=centers_coord,
              inst_cluster_id=inst_cluster_id,
              labels=labels),
            }

        return ans
