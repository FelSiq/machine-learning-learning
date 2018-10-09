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
		-------------------------------------------
	"""
	def __euclideandist__(inst_a, inst_b)
		return (sum((inst_a - inst_b)**2.0))**0.5

	def run(dataset, radius, minpts):
		pass
	

if __name__ == "__main__":
	import sys

	if len(sys.argv) < 4:
		print("usage:", sys.argv[0], 
			"<data_filepath> <radius> <min_pts>",
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

