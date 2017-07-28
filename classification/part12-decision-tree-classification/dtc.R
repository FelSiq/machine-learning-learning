# Get the dataset
dataset <- read.csv('/home/felipe/Documentos/Machine Learning A-Z/Part 3 - Classification/Section 14 - Logistic Regression/Social_Network_Ads.csv')

# Check for categorical/hierarchical parameters
dataset <- dataset[3:ncol(dataset)]
dataset$Purchased <- as.factor(dataset$Purchased) # Needed to rpart work correctly

# Feature scaling? No need with the classification tree
# dataset[, -ncol(dataset)] <- scale(apply(dataset[, -ncol(dataset)], 2, as.numeric))

# Split the dataset into test set and train set
library(caTools)
datasplit <- sample.split(dataset$Purchased, SplitRatio = 0.75)
train_set <- subset(dataset, datasplit)
test_set <- subset(dataset, !datasplit)

# Fit the model
library(rpart)
classifier <- rpart(
	formula = Purchased ~ .,
	data = train_set,
	control = rpart.control(minsplit = 1))

# Predict the test set
prediction <- predict(
	object = classifier, 
	newdata = test_set, 
	type = 'class') # To predict a factor vector, not a nx2 matrix with the probabilities
# Confusion matrix
table(test_set$Purchased, prediction)

# Plot the decision tree 
plot(classifier)
text(classifier)

# =============================================
# Disclaimer: All the plotting code below is not mine.
# Font: Machine Learning A-Z™: Hands-On Python & R In Data Science - kernel_svm.R source code 
# =============================================
# Caution: use this only with feature scaling.
library(ElemStatLearn)
set = train_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set, type = 'class')
plot(set[, -3],
     main = 'Decision Tree (Training set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))