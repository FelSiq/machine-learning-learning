# Get the dataset
dataset <- read.csv('/home/felipe/Documentos/Machine Learning A-Z/Part 3 - Classification/Section 14 - Logistic Regression/Social_Network_Ads.csv')
dataset <- dataset[3:ncol(dataset)]

# Feature scaling
dataset[, 1:(ncol(dataset) - 1)] <- scale(apply(dataset[, 1:(ncol(dataset) - 1)], 2, as.numeric))

# Split the dataset into test set and train set
library(caTools)
datasplit <- sample.split(dataset$Purchased, SplitRatio = 0.75)
train_set <- subset(dataset, datasplit)
test_set <- subset(dataset, !datasplit)

# Visualize the dataset
library(ggplot2)
ggplot() + 
	geom_point(aes(
			x = subset(train_set, train_set$Purchased == TRUE)$Age, 
			y = subset(train_set, train_set$Purchased == TRUE)$EstimatedSalary), 
		color = 'blue') +
	geom_point(aes(
			x = subset(train_set, train_set$Purchased == FALSE)$Age, 
			y = subset(train_set, train_set$Purchased == FALSE)$EstimatedSalary), 
		color = 'red') +
	xlab('Age') +
	ylab('Estimated Salary') +
	ggtitle('Customers Scatter Plot') +
	xlim(min(train_set$Age) * 1.25, max(train_set$Age) * 1.25) +
	ylim(min(train_set$EstimatedSalary) * 1.25, max(train_set$EstimatedSalary) * 1.25)

# K-NN is a lazy algorithm, there's no explicit model. Then, the training
# step and prediction step are the same.
library(class)
prediction <- knn(
	train = train_set[, -ncol(train_set)], 
	test = test_set[, -ncol(test_set)],
	cl = train_set$Purchased,
	k = 5)

# Confusion matrix
conf_mat <- table(test_set$Purchased, prediction)
conf_mat

# K-NN probabilities
prediction_prob <- knn(train = train_set[, -ncol(train_set)], 
	test = test_set[, -ncol(test_set)],
	cl = train_set$Purchased,
	k = 5,
	prob = TRUE)