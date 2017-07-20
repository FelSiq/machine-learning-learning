# Get the dataset
import pandas as pd
dataset = pd.read_csv('/home/felipe/Documentos/Machine Learning A-Z/Part 2' + 
	' - Regression/Section 7 - Support Vector Regression (SVR)/Position_Salaries.csv')
ind_param = dataset.iloc[:, 1:2].values
dep_param = dataset.iloc[:,  -1].values

# No missing values neither categorical data.

# Can't split the dataset into train and test set, due to it's small size

# Feature scale is a must in Support Vector Regression Class in Python Package
from sklearn.preprocessing import StandardScaler as sklp_ss
ind_scaler = sklp_ss()
ind_param = ind_scaler.fit_transform(ind_param.reshape(-1, 1))
dep_scaler = sklp_ss()
dep_param = dep_scaler.fit_transform(dep_param.reshape(-1, 1))

# Construct a initial kernel model
from sklearn.svm import SVR as svr
regressor = svr(kernel = 'rbf')
regressor.fit(ind_param, dep_param)

# Make a predicition
import numpy as np
# First, create a numpy array of the wanted input values
input_prediction = np.array([[6.5]])
# Then, transform it input array to the scale of the model
# BEWARE: use the independent (x) parameter scaler!
input_prediction = ind_scaler.transform(input_prediction)
# Make a prediction with the SVR model
predictions = regressor.predict(input_prediction)
# Now do a inverse transformation, in order to inperpret the result
# CAUTION: use the dependent (f(x)) parameter scaler!
predictions = dep_scaler.inverse_transform(predictions)
# One line code: predictions = dep_scaler.inverse_transform(regressor.predict(ind_scaler.transform(np.array([[6.5]]))))
print predictions

# Verify the performance of the SVR model
import matplotlib.pyplot as plt
plt.scatter(ind_param, dep_param, color = 'blue')
plt.plot(ind_param, regressor.predict(ind_param), color = 'red')
plt.title('True values vs Prediction')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()
