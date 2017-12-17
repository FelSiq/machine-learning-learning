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
classifier_e1071_lin <- svm(
	formula = Purchased ~ ., data = train_set, type = 'C-classification', kernel = 'linear')
classifier_e1071_gau <- svm(
	formula = Purchased ~ ., data = train_set, type = 'C-classification', kernel = 'radial')
classifier_e1071_pol <- svm(
	formula = Purchased ~ ., data = train_set, type = 'C-classification', kernel = 'polynomial', degree = 3, coef0 = 0.0)
classifier_e1071_sig <- svm(
	formula = Purchased ~ ., data = train_set, type = 'C-classification', kernel = 'sigmoid', coef0 = 0.0)

# Now with kernlab
# kpar=list(sigma=0.05)
library(kernlab)
classifier_kernlab_lin <- ksvm(x = Purchased ~ ., data = train_set,	type = 'C-svc',	kernel = 'vanilladot')
classifier_kernlab_pol <- ksvm(x = Purchased ~ ., data = train_set,	type = 'C-svc',	kernel = 'polydot')
classifier_kernlab_rbf <- ksvm(x = Purchased ~ ., data = train_set,	type = 'C-svc',	kernel = 'rbfdot')
classifier_kernlab_tan <- ksvm(x = Purchased ~ ., data = train_set,	type = 'C-svc',	kernel = 'tanhdot')
classifier_kernlab_lap <- ksvm(x = Purchased ~ ., data = train_set,	type = 'C-svc',	kernel = 'laplacedot')
classifier_kernlab_bes <- ksvm(x = Purchased ~ ., data = train_set,	type = 'C-svc',	kernel = 'besseldot')
classifier_kernlab_anv <- ksvm(x = Purchased ~ ., data = train_set,	type = 'C-svc',	kernel = 'anovadot')
classifier_kernlab_spl <- ksvm(x = Purchased ~ ., data = train_set,	type = 'C-svc',	kernel = 'splinedot')

# Predict the test set
prediction_e1071_lin <- predict(classifier_e1071_lin, newdata = test_set)
prediction_e1071_gau <- predict(classifier_e1071_gau, newdata = test_set)
prediction_e1071_pol <- predict(classifier_e1071_pol, newdata = test_set)
prediction_e1071_sig <- predict(classifier_e1071_sig, newdata = test_set)

prediction_kernlab_lin <- predict(classifier_kernlab_lin, newdata = test_set)
prediction_kernlab_pol <- predict(classifier_kernlab_pol, newdata = test_set)
prediction_kernlab_rbf <- predict(classifier_kernlab_rbf, newdata = test_set)
prediction_kernlab_tan <- predict(classifier_kernlab_tan, newdata = test_set)
prediction_kernlab_lap <- predict(classifier_kernlab_lap, newdata = test_set)
prediction_kernlab_bes <- predict(classifier_kernlab_bes, newdata = test_set)
prediction_kernlab_anv <- predict(classifier_kernlab_anv, newdata = test_set)
prediction_kernlab_spl <- predict(classifier_kernlab_spl, newdata = test_set)

# Confusion matrix (exactly same result on both packages!)
table(test_set$Purchased, prediction_e1071_lin)
table(test_set$Purchased, prediction_e1071_gau)
table(test_set$Purchased, prediction_e1071_pol)
table(test_set$Purchased, prediction_e1071_sig)

table(test_set$Purchased, prediction_kernlab_lin)
table(test_set$Purchased, prediction_kernlab_pol)
table(test_set$Purchased, prediction_kernlab_rbf)
table(test_set$Purchased, prediction_kernlab_tan)
table(test_set$Purchased, prediction_kernlab_lap)
table(test_set$Purchased, prediction_kernlab_bes)
table(test_set$Purchased, prediction_kernlab_anv)
table(test_set$Purchased, prediction_kernlab_spl)

# =============================================
# Disclaimer: All the plotting code below is not mine.
# Font: Machine Learning A-Z™: Hands-On Python & R In Data Science - kernel_svm.R source code 
# =============================================
library(ElemStatLearn)
classifier <- classifier_kernlab_spl
set = train_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'Kernel SVM (Training set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))