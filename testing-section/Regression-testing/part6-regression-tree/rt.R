# ========================================
# Decision Tree Regressor is not a continuous regressor, 
# it is a interval-wise (discrete) regressor instead.
# ========================================
# get the dataset
dataset <- read.csv('/home/felipe/Documentos/Machine Learning A-Z/Part 2 - Regression/Section 8 - Decision Tree Regression/Position_Salaries.csv')
dataset <- dataset[, 2:length(dataset)]
# ========================================
# No missing values neither categorical values
# ========================================
# No need for split into train and test sets
# ========================================
# No need for feature scalling? No! Because it's a Decision Tree model!
# ========================================
# Fit (construct) the Decision Tree Regressor
library(rpart)
regressor <- rpart(
	formula = Salary ~ .,
	data = dataset,
	control = rpart.control(minsplit = 1))
# ========================================
# Verify (Plot the results) with high definition
library(ggplot2)
level_grid <- seq(min(dataset$Level), max(dataset$Level), 0.01) 
ggplot() +
	geom_point(aes(x = dataset$Level, y = dataset$Salary), color = 'blue') +
	geom_line(aes(x = level_grid, y = predict(object = regressor, newdata = data.frame(Level = level_grid))), color = 'red') +
 	ggtitle('Level vs Salary') +
	xlab('Level') +
	ylab('Salary')
# ========================================
# Predict new values
predict(regressor, data.frame(Level = 6.5))
# ========================================