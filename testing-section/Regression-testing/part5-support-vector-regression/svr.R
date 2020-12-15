# Get the dataset
dataset <- read.csv('/home/felipe/Documentos/Machine Learning A-Z/Part 2 - Regression/Section 7 - Support Vector Regression (SVR)/Position_Salaries.csv')
# New outlier sample: (to keep 'CEO' level on SVR model)
dataset$Position <- as.character(dataset$Position)
dataset[11, ] <- data.frame(Position = 'Outlier', Level = 11, Salary = 3e+06)
dataset$Position[11] <- 'Outlier'
dataset$Position <- as.factor(dataset$Position)

# Remove the redundant information column
dataset <- dataset[, 2:3]

# Construct the SVR model
library(e1071)
regressor <- svm(
	formula = Salary ~ ., 
	data = dataset, 
	scale = TRUE, # Value Scalling is on by default, but just to be sure 
	type = 'eps-regression') # eps -> Regression, C -> classification

# Verify the performance
library(ggplot2)
ggplot() +
	geom_point(aes(x = dataset$Level, y = dataset$Salary), color = 'blue') +
	geom_line(aes(x = dataset$Level, y = predict(object = regressor, newdata = dataset)), color = 'red') +
	ggtitle('Salary vs Level') +
	xlab('Level') +
	ylab('Salary')

# Predict a new value
predict(regressor, data.frame(Level = 6.5))