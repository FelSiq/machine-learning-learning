# Get the dataset
dataset <- read.csv('/home/felipe/Documentos/Machine Learning A-Z/Part 3 - Classification/Section 14 - Logistic Regression/Social_Network_Ads.csv')

# No missing values

# Check for categorical/hierarchical parameters
dataset <- dataset[2:ncol(dataset)]
dataset$Gender <- factor(dataset$Gender, 
	levels = unique(dataset$Gender), 
	labels = seq(0, length(unique(dataset$Gender)) - 1, 1))

# Feature scaling
dataset[, 1:(ncol(dataset) - 1)] <- scale(apply(dataset[, 1:(ncol(dataset) - 1)], 2, as.numeric))

# Split the dataset into test set and train set
library(caTools)
datasplit <- sample.split(dataset$Purchased, SplitRatio = 0.75)
train_set <- subset(dataset, datasplit)
test_set <- subset(dataset, !datasplit)

# Fit the model
classifier <- glm(formula = Purchased ~ Age + EstimatedSalary, 
	family = binomial, # Because it's a logist regression 
	data = train_set)
# Seens like 'Gender' parameter is not a very influent parameter

# Predict the test set
prediction_prob <- predict(object = classifier, 
	type = 'response', 
	newdata = test_set)

# Threshold: 0.5
prediction_label <- ifelse(prediction_prob >= 0.5, 1, 0)

# Confusion matrix
conf_mat <- table(test_set$Purchased, prediction_label)