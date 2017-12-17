# Data preprocessing
dataset <- read.csv('/home/felipe/Documentos/Machine Learning A-Z/Part 1 - Data Preprocessing/Data.csv')

# Missing data
fillMV <- function(vec, func) ifelse(is.na(vec), func(vec, na.rm = TRUE), vec) 
dataset$Age <- fillMV(dataset$Age, median)
dataset$Salary <- fillMV(dataset$Salary, median)

# Discretize categorical values
discretize <- function(vec) factor(vec,
	levels = unique(vec),
	labels = seq(0, length(unique(vec)) - 1, 1))
dataset$Country <- discretize(dataset$Country)
dataset$Purchased <- discretize(dataset$Purchased)

# Split the train set and the test set
library(caTools)
datasplit <- sample.split(dataset$Purchased, SplitRatio = 0.7)
training_set <- subset(dataset, datasplit == TRUE)
test_set <- subset(dataset, datasplit == FALSE)

# Scalling parameters
# Normalization:
#	new_value = (old_value - min)/(max - min)
# Standartization:
#	new_value = (old_value - mean)/(standard_devitation)
training_set[ , c(2,3)] <- scale(apply(training_set[ , c(2,3)], 2, as.numeric))
test_set[ , c(2,3)] <- scale(apply(training_set[ , c(2,3)], 2, as.numeric))