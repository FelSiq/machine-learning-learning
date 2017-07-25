# Get the dataset
dataset <- read.csv('/home/felipe/Documentos/Machine Learning A-Z/Part 2 - Regression/Section 5 - Multiple Linear Regression/50_Startups.csv')

# No missing values.

# Found categorical values (R does not need dummy variables, through, just encodification)
dataset$State <- factor(dataset$State, 
	levels = unique(dataset$State), 
	labels = seq(1, length(unique(dataset$State)), 1))

# Split the dataset into a train and test set
library(caTools)
datasplit <- sample.split(dataset$State, SplitRatio = 0.8)
training_set <- subset(dataset, datasplit == TRUE)
test_set <- subset(dataset, datasplit == FALSE)

# R's MLR package already take care feature scaling for us.

# Create a model of MLR fitted to our dataset
# regressor <- lm(
#	formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State, 
#	data = training_set)
regressorMult <- lm(
	formula = Profit ~ ., 
	data = training_set)

# Based on the summary() information, only the R.D.Spend is a
# important prediction variable. In fact, we could even write the
# linear regression as:
regressorSing <- lm (
	formula = Profit ~ R.D.Spend,
 	data = training_set)
# And that would have the same prediction power. (Test it! [ ])

# Predict the values with the regressor
predictionsMult <- predict(object = regressorMult, newdata = test_set)
predictionsSing <- predict(object = regressorSing, newdata = test_set)

# Plot the results
library(ggplot2)
ggplot() + 
	geom_point(aes(x = test_set$Profit, y = predictionsMult), 
		colour = 'blue') +
	geom_point(aes(x = test_set$Profit, y = predictionsSing), 
		colour = 'green') +
	geom_line(aes(x = c(0, max(test_set$Profit) * 1.1), y = c(0, max(test_set$Profit) * 1.1)),
		colour = 'red') +
	ggtitle('TrueValue vs PredValue') +
	xlab('TrueProfit') +
	ylab('PredictionValue')

# Optimal model with Backward Elimination?
# Remember: R's functions already take care of the dummy variables.
regressorOpt <- lm(formula = Profit ~ ., data = training_set)
summary(regressorOpt)
regressorOpt <- lm(formula = Profit ~ R.D.Spend + Marketing.Spend + State, data = training_set)
summary(regressorOpt)
regressorOpt <- lm(formula = Profit ~ R.D.Spend + Marketing.Spend, data = training_set)
summary(regressorOpt)
regressorOpt <- lm(formula = Profit ~ R.D.Spend, data = training_set)
summary(regressorOpt)

# This turned into a simple linear regression
# Even more, this is exactly the same model made in #30 code line.