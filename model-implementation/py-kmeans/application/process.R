library(stats)

numClusters <- c(2, 3, 5, 7, 10, 15)

data <- read.table('./data.out')

for (n in numClusters) {
	model <- stats::kmeans(data, centers=n)
	newAudio <- model$centers[model$cluster]
	sink(paste('./newAudio', n, sep=''))
	for (m in newAudio)
		cat(m, '\n')
	sink(NULL)
}