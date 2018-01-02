# ----------------------------------------------
# INFORMATION (ENN - Edited Nearest Neighbors)
# ----------------------------------------------
# This is a simple implementation of ENN Noise Filter. Given a dataset, 
# it returns the index of which instances are possible noises, based on
# if the majority of the k-nearest neighbors for each instance have different classes.
# ----------------------------------------------

ENN <- function(data, k = 3) {
	x <- data[-which(colnames(data) == 'Class')]
	y <- data$Class

	possibleNoises <- vector()

	for (i in 1:nrow(data)) {
		dist <- vector()
		for (j in 1:nrow(data)) {
			dist[j] <- sum((x[i,] - x[j,])^2.0)^0.5 
		}
		# The first index is not considered because the distance is always 0.0, when i == j.
		# I don't put a 'ifelse' on the 'for' above because this would aggregate unnecessary 
		# extra computational cost. Then, it is simpler and cheaper just ignore the first index. 
		knn <- sort.list(dist)[2:(k + 1)]
		
		equalClasses <- sum(y[i] == y[knn])

		if (equalClasses < k/2.0) {
			possibleNoises <- c(possibleNoises, i)
		}
	}
	return (possibleNoises)
}


# ----------------------------------------------
# INFORMATION (AENN - All-k Edited Nearest Neighbors)
# ----------------------------------------------
# AENN apply ENN for all integer i between 1 and k, and then remove
# all instances considered noise for any i.
# ----------------------------------------------

AENN <- function(data, k = 3) {
	noises <- vector()
	for (i in 1:k) {
		noises <- c(noises, ENN(data, i))
	}
	return(data[-noises,])
}