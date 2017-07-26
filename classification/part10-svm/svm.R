# Get the dataset
dataset <- read.csv('/home/felipe/Documentos/Machine Learning A-Z/Part 3 - Classification/Section 14 - Logistic Regression/Social_Network_Ads.csv')

# Check for categorical/hierarchical parameters
dataset <- dataset[3:ncol(dataset)]

# Feature scaling
dataset[, 1:(ncol(dataset) - 1)] <- scale(apply(dataset[, 1:(ncol(dataset) - 1)], 2, as.numeric))

# Split the dataset into test set and train set
library(caTools)
datasplit <- sample.split(dataset$Purchased, SplitRatio = 0.75)
train_set <- subset(dataset, datasplit)
test_set <- subset(dataset, !datasplit)

# Fit the model with e1071
library(e1071)
classifier_e1071 <- svm(formula = Purchased ~ .,
	data = train_set,
	type = 'C-classification',
	kernel = 'linear')

# Now with kernlab
library(kernlab)
classifier_kernlab <- ksvm(x = Purchased ~ .,
	data = train_set,
	type = 'C-svc',
	kernel = 'vanilladot')

# Predict the test set
prediction_e1071 <- predict(classifier_e1071, newdata = test_set)
prediction_kernlab <- predict(classifier_kernlab, newdata = test_set)

# Confusion matrix (exactly same result on both packages!)
table(test_set$Purchased, prediction_e1071)
table(test_set$Purchased, prediction_kernlab)