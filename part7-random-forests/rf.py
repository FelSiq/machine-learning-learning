# Get the dataset
import pandas as pd
dataset = pd.read_csv('/home/felipe/Documentos/Machine Learning A-Z/' + 
	'Part 2 - Regression/Section 9 - Random Forest Regression/Position_Salaries.csv')
# Outlier treatment? 
test_dataframe = pd.DataFrame({
	'Position':	['Outlier'],
	'Level':	[11],
	'Salary':	[3e+06]
	})
# Concatenate the new outlier sample into the original dataset
dataset = pd.concat([dataset, test_dataframe])
dataset = dataset.reindex_axis(['Position', 'Level', 'Salary'], axis = 1)

# split into dependent array and independent matrix
ind_var = dataset.iloc[:, 1:-1].values
dep_var = dataset.iloc[:, -1].values

# Verify Missing values
# No Missing values.

# Check for categorical/hierarchical values
# No categorical data on the final dataset.

# Check if data scaling is necessary with RF algorithm
# No scaling on RF.

# Split the dataset into train and test sets
# Not needed, as the dataset is very small.

# Fit a model with the train set
from sklearn.ensemble import RandomForestRegressor as skl_rfr
regressor10 = skl_rfr(n_estimators = 10, criterion = 'mse')
regressor100 = skl_rfr(n_estimators = 100, criterion = 'mse')
regressor1000 = skl_rfr(n_estimators = 1000, criterion = 'mse')
regressor10.fit(ind_var, dep_var)
regressor100.fit(ind_var, dep_var)
regressor1000.fit(ind_var, dep_var)

# Test the model with the test set
regressor10.predict(6.5)
regressor100.predict(6.5)
regressor1000.predict(6.5)

# Plot result, with high definition, to verify the performance
import matplotlib.pyplot as plt
import numpy as np
ind_var_grid = np.arange(min(ind_var), max(ind_var), 0.01)
ind_var_grid = ind_var_grid.reshape(len(ind_var_grid), 1)
plt.scatter(ind_var, dep_var, color = 'blue')
plt.plot(ind_var_grid, regressor10.predict(ind_var_grid), color = 'red')
plt.plot(ind_var_grid, regressor100.predict(ind_var_grid), color = 'red')
plt.plot(ind_var_grid, regressor1000.predict(ind_var_grid), color = 'red')
plt.title('Level vs Salary')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()