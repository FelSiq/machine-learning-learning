# Get the dataset
dataset <- read.csv('/home/felipe/Documentos/Machine Learning A-Z/Part 2 - Regression/Section 6 - Polynomial Regression/Position_Salaries.csv')
dataset <- dataset[, 2:3]

# Verify, visually, the dataset
# plot(dataset)

# Fit a liear model (for comparison purposes)
lin_reg <- lm(formula = Salary ~ Level, data = dataset)
summary(lin_reg)

# Fit the polynomial regressor model =====================
# -		Model degree: 2 
dataset$Level2 <- dataset$Level ^ 2
pol_reg_degree2 <- lm(formula = Salary ~ ., data = dataset)
summary(pol_reg_degree2) # Better!

# -		Model degree: 4
dataset$Level3 <- dataset$Level ^ 3
dataset$Level4 <- dataset$Level ^ 4
pol_reg_degree4 <- lm(formula = Salary ~ ., data = dataset)
summary(pol_reg_degree4) # Even Better!

# -		Model degree: 8
dataset$Level5 <- dataset$Level ^ 5
dataset$Level6 <- dataset$Level ^ 6
dataset$Level7 <- dataset$Level ^ 7
dataset$Level8 <- dataset$Level ^ 8
pol_reg_degree8 <- lm(formula = Salary ~ ., data = dataset)
summary(pol_reg_degree8) # Fucked up model? (Overfitted?)

# Plot the results (with high definition) ================
library(ggplot2)
ggplot() + 
	geom_point(aes(x = dataset$Level, y = dataset$Salary), 
		colour = 'blue') +
	geom_line(aes(x = dataset$Level, y = predict(object = lin_reg, newdata = dataset)),
		colour = 'red') +
	geom_line(aes(x = dataset$Level, y = predict(object = pol_reg_degree2, newdata = dataset)),
		colour = 'darkred') +
	geom_line(aes(x = dataset$Level, y = predict(object = pol_reg_degree4, newdata = dataset)),
		colour = 'purple') +
	geom_line(aes(x = dataset$Level, y = predict(object = pol_reg_degree8, newdata = dataset)),
		colour = 'cyan') +
	ggtitle('Salary vs Experience') +
	xlab('Years of Experience') +
	ylab('Salary')

# Predict new values =====================================
