knn <- function(dataset, query, k=3) {
	dtcols <- ncol(dataset)
	model <- list()

	model$index <- seq(1, nrow(dataset), 1)
	model$dist <- apply(dataset, 1, function(x) (sum((as.numeric(x[1:(dtcols - 1)]) - query)^2))^0.5)

	nn <- model$index[sort.list(model$dist, decreasing = FALSE)]

	votes <- dataset[nn, dtcols][1:k]
	
	ret <- list()
	classes <- unique(dataset[[dtcols]])

	ret$classes <- as.vector(classes)

	for (i in 1:length(classes)) {
		ret$votes[i] <- sum(votes == classes[i]) 
	}

	ret$result <- as.character(classes[sort.list(ret$votes, decreasing = TRUE)[1]])

	return (ret)
}

knn(iris, c(5.9543, 3.242, 5.1, 1.9423), 7)