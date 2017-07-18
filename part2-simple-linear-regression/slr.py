# Data preprocessing templace:
# 1. Import the libraries
# 2. Import the dataset
# 3. Check for missing values
# 4. Encode discrete (categorical) values
# 5. CHeck if feature scaling is necessary
# 6. Split the dataset in training set and test set

# Section one: import the needed libraries
import pandas as pd # This package is great for data handling in python

# Section two: import the dataset
dataset = pd.read_csv("/home/felipe/Documentos/Machine Learning A-Z/"
	"Part 2 - Regression/Section 4 - Simple Linear Regression/Salary_Data.csv")
p_input = dataset.iloc[:, :-1].values # Independent variable 
p_output = dataset.iloc[:, -1].values # Dependent variable

# Section three: check for missing data
# No missing data!

# Section four: Encode categorical values
# No categorical values!

# Section five: check if feature scaling is necessary
# It can be, depending of the package used.

# Section six: split the dataset into train and test set
from sklearn.model_selection import train_test_split as tts
p_input_train, p_input_test, p_output_train, p_output_test = tts(p_input, p_output, test_size = 0.33)

# Fitting Simple Linear Regression (SLR) into the training set
from sklearn.linear_model import LinearRegression as lr
regressor = lr() # Call the constructor method of the sklearn linear regressor
regressor.fit(p_input_train, p_output_train) # Construct the model with the train set

# Testing the model with the test set
predictions = regressor.predict(p_input_test) # Make predictions

# Plot in the train set
import matplotlib.pyplot as plt
plt.scatter(p_input_train, p_output_train, color = 'red', label = 'Train Set')
plt.scatter(p_input_test, p_output_test, color = 'green', label = 'Test Set')
# plt.ticklabel_format(style = 'sci', axis = 'y', scilimits = (0, 0))
plt.plot(p_input_train, regressor.predict(p_input_train), color = 'blue')
plt.title('Salary vs Experience')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()

# Be careful! When plotting the TEST SET, the linear regression line
# is the exactly same line of the previous plot!