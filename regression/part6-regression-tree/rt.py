# Get the dataset
import pandas as pd
dataset = pd.read_csv('/home/felipe/Documentos/Machine Learning A-Z/' + 
	'Part 2 - Regression/Section 8 - Decision Tree Regression/Position_Salaries.csv')
# =================================
# No missing data neither categorical values
# =================================
# Split the dataset into the dependent and independent variables
ind_param = dataset.iloc[:, 1:-1].values
dep_param = dataset.iloc[:,  -1].values
# =================================
# No split into test and train set with this dataset.
# =================================
# Outlier treatment? (Not needed)
# =================================
# Feature scalling? (Not needed)
# =================================
# Built (fit) the decision tree regression model
from sklearn.tree import DecisionTreeRegressor as sklt_dtr
# mse = 'Mean Squared Error'
regressor = sklt_dtr(criterion = 'mse')
regressor.fit(ind_param.reshape(-1, 1), dep_param.ravel())
# =================================
# Plot (verify) the results
# BEWARE: plot in high definition, or else plotting will be biased.
import matplotlib.pyplot as plt
import numpy as np
ind_param_grid = np.arange(min(ind_param), max(ind_param), 0.01)
ind_param_grid = ind_param_grid.reshape(len(ind_param_grid), 1)
plt.scatter(ind_param, dep_param, color = 'blue')
plt.plot(ind_param_grid, regressor.predict(ind_param_grid), color = 'red')
plt.title('Level vs Salary')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()
# =================================
# Predict new values
regressor.predict(6.5)
# =================================