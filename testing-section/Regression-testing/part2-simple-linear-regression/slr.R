# Data preprocessing templace:
# 1. Import the dataset
# 2. Check for missing values
# 3. Encode discrete (categorical) values
# 4. CHeck if feature scaling is necessary
# 5. Split the dataset in training set and test set

# Section one: import the dataset
dataset <- read.csv("/home/felipe/Documentos/Machine Learning A-Z/Part 2 - Regression/Section 4 - Simple Linear Regression/Salary_Data.csv")

# A simple line
# y = a + bx
# a: linear coef
# b: angular coef (slope)

# Section 2: No missing values!

# Section 3: No categorical values!

# Section 4: No scaling necessary, on this case.

# Section 5: Split the dataset into training and test set
library(caTools)
split <- sample.split(dataset$YearsExperience, SplitRatio = 2/3)
train_set <- subset(dataset, split == TRUE)
test_set <- subset(dataset, split == FALSE)

# Construct the Linear Regression Model
regressor <- lm(formula = Salary ~ YearsExperience, data = train_set)
summary(regressor)

# Using the linear model into the test set
predictions <- predict(object = regressor, newdata = test_set)

# Plot the results
library(ggplot2)
ggplot() + 
	geom_point(aes(x = train_set$YearsExperience, y = train_set$Salary), 
		colour = 'blue') +
	geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary), 
		colour = 'green') +
	geom_line(aes(x = train_set$YearsExperience, y = predict(object = regressor, newdata = train_set)),
		colour = 'red') +
	ggtitle('Salary vs Experience') +
	xlab('Years of Experience') +
	ylab('Salary')