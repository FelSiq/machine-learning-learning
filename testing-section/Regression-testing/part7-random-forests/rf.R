# Get the dataset
dataset <- read.csv('/home/felipe/Documentos/Machine Learning A-Z/Part 2 - Regression/Section 9 - Random Forest Regression/Position_Salaries.csv')

# New outlier sample: (to keep 'CEO' level on SVR model)
dataset$Position <- as.character(dataset$Position)
dataset[11, ] <- data.frame(Position = 'Outlier', Level = 11, Salary = 3e+06)
dataset$Position[11] <- 'Outlier'
dataset$Position <- as.factor(dataset$Position)
dataset <- dataset[, 2:3]

# No missing values neither categorical/hierarchical parameters
# No need to split into train and test sets.
# No need for data scaling.

# Fit the model on the train set
library(randomForest)
create_model <- function(dataset, ntree = 10) {
	return (randomForest(x = dataset[1:(ncol(dataset) - 1)], y = dataset[, ncol(dataset)], ntree = ntree))
}

regressor10 <- create_model(dataset, 10)
regressor100 <- create_model(dataset, 100)
regressor1000 <- create_model(dataset, 1000)

# Plot the results with high definition
library(ggplot2)
level_grid <- seq(min(dataset$Level), max(dataset$Level), 0.01) 
ggplot() +
	geom_point(aes(x = dataset$Level, y = dataset$Salary), color = 'blue') +
	geom_line(aes(x = level_grid, y = predict(object = regressor10, newdata = data.frame(Level = level_grid))), color = 'red') +
	geom_line(aes(x = level_grid, y = predict(object = regressor100, newdata = data.frame(Level = level_grid))), color = 'green') +
	geom_line(aes(x = level_grid, y = predict(object = regressor1000, newdata = data.frame(Level = level_grid))), color = 'purple') +
	ggtitle('Salary vs Level') +
	xlab('Level') +
	ylab('Salary')

# Predict a new value
predict(object = regressor10, data.frame(Level = 6.5))
predict(object = regressor100, data.frame(Level = 6.5))
predict(object = regressor1000, data.frame(Level = 6.5))