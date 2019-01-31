"""
	Useful link: http://www.cs.fsu.edu/~ackerman/CIS5930/notes/DBSCAN.pdf
	Title: " DBSCAN: Density-Based Spatial Clustering of Applications with Noise"
	Presented by Wondong Lee
	Written by M.Ester, H.P.Kriegel, J.Sander and Xu. 
"""

from pandas import read_csv
from numpy import array


class Dbscan():
    """
		Dbscan is an unsupervised machine learning
		algorithm which clusters points with an ar-
		bitrary distribution shape.

		For each point in dataset, Dbscan check the
		density of its neighborhood. It a point is
		a high-density neighborhood point (i.e. it
		has a MINIMAL NUMBER of neighbors), then it
		is considered a CORE POINT. 

		Otherwise, if the point does not have a MI-
		NIMAL NUMBER of neighbors, the point is a
		low-density neighborhood point.

		If a point is a low-density neighborhood point,
		then two scenarios can happen:
		- Either that point is in the neighborhood of a
		CORE POINT, and therefore is considered a BORDER
		POINT 
		- Otherwise, it is considered a OUTLIER POINT.

		Dbscan takes two main parameters:
			epsilon/radius : the radius of each
				point, which decides which
				points are it's neighbors.
			minpts : the minimal number of points
				in the neighbor of a given po-
				int in order to not classify
				it as a outlier (low density
				point).
		-------------------------------------------
		Example: Let be
			C : Core point
			B : Border point
			O : Outlier point

		For a given radius "e" and MINPOINTS=3, we
		may have:
		o-------------------------------o
		|		     B		|
		|		C           O	|
		|	      C C       O	|
		|    O          		|
		|	    B     B		|
		|				|
		o-------------------------------o
		-------------------------------------------
		DBSCAN GENERAL CONCEPTS:
		There are few core concepts you must master
		before truly understanding the DBSCAN algo-
		rithm.
		-------------------------------------------
		Directly Density-reachability concept:
		B is said directly density-reachable from A
		if A is a CORE POINT and B is in neighborhood
		of A.

		Directly Dentisity-reachability is assymetric: if B
		is directly density-reachable from A, it does
		NOT mean that A is directly density-reachable
		from B, as B can be a non-core point.
		-------------------------------------------
		Density-reachable concept:
		A point p is density-reachable from the point q
		if there's exists such a chain of points
		c_1, ..., c_n such that

			p = c1
			q = cn
			c_i is directly density-reachable from c_(i+1)
		-------------------------------------------
		Density-connectivity concept:
		Two pair of points is said density-connected if
		they're commonly density-reachable from and to
		each other.

		A is density-reachable from B 
				^	
		B is density-reachable from A
				<=> 
		A is Density-connected with B
				<=> 
		B is Density-connected with A

		From the definition, is easy to see the the
		density-connectivity concept, different from
		the density-reachable concept, is symmetric.
		-------------------------------------------
	"""

    def __euclideandist__(inst_a, inst_b):
        return (sum((inst_a - inst_b)**2.0))**0.5

    def run(dataset, radius, minpts):

        possible_noises = set()
        cluster_id = array([-1] * dataset.shape[0])
        node_core = array([False] * dataset.shape[0])
        counter = 0

        equiv_cluster_ids = {}

        # Ultimately all instances must be checked
        for inst_id in range(dataset.shape[0]):

            cur_neighborhood = set()

            for neighbor_id in range(dataset.shape[0]):
                if Dbscan.__euclideandist__(\
                 dataset[inst_id,:], \
                 dataset[neighbor_id,:]) <= radius:

                    cur_neighborhood.update({neighbor_id})

            if len(cur_neighborhood) >= minpts:
                # New core point detected
                node_core[inst_id] = True

                if cluster_id[inst_id] == -1:
                    # This core point does not belong
                    # to any previously created cluster, so
                    # create a new one
                    cluster_id[inst_id] = counter
                    neighbor_cluster_id = counter
                    counter += 1
                else:
                    # This core point is density-reachable
                    # to another core point in an existing
                    # cluster.
                    neighbor_cluster_id = cluster_id[inst_id]

                # "Paint" current neighbors with the same
                # cluster id of this current instance.
                for neighbor_id in cur_neighborhood:
                    if node_core[neighbor_id]:
                        # If this ever happens, it means that
                        # we have two core nodes of the same
                        # cluster, but with cluster_ids distincts.
                        # In other words, both cluster_id repre-
                        # sents the same cluster.
                        if cluster_id[inst_id] not in equiv_cluster_ids:
                            equiv_cluster_ids[cluster_id[inst_id]] =\
                             cluster_id[neighbor_id]
                        else:
                            equiv_cluster_ids[cluster_id[neighbor_id]] =\
                             equiv_cluster_ids[cluster_id[inst_id]]

                    cluster_id[neighbor_id] = neighbor_cluster_id

            elif cluster_id[inst_id] == -1:
                # Current instance is not a CORE point and
                # is still not painted (i.e. it is not classified
                # as a BORDER point until now), so it can be a
                # possible noisy instance.
                possible_noises.update({inst_id})

        # For each instance marked as a possible noise,
        # recheck if it is not "colored". If not, then
        # the instance is a true noise.
        noise_id = set()
        for inst_id in possible_noises:
            if cluster_id[inst_id] == -1:
                noise_id.update({inst_id})

        for inst_id in range(dataset.shape[0]):
            cur_inst_id = cluster_id[inst_id]
            if cluster_id[inst_id] in equiv_cluster_ids:
                cluster_id[inst_id] = equiv_cluster_ids[cur_inst_id]

        ans = {
            "cluster_id": cluster_id,
            "noise": noise_id,
        }

        return ans


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import sys

    if len(sys.argv) < 4:
        print("usage:", sys.argv[0], "<data_filepath> <radius> <min_pts>",
              "[-sep data_separator, default to \",\"]")
        exit(1)

    try:
        sep = sys.argv[1 + sys.argv.index("-sep")]
    except:
        sep = ","

    dataset = read_csv(sys.argv[1], sep=sep)

    ans = Dbscan.run(
        dataset=dataset.values,
        radius=float(sys.argv[2]),
        minpts=int(sys.argv[3]))

    if ans:
        print("Results:")
        for item in ans:
            print(item, ":", ans[item])

        if dataset.shape[1] == 2:
            plt.scatter(
                x=dataset.iloc[:, 0],
                y=dataset.iloc[:, 1],
                c=ans["cluster_id"])
            plt.show()
