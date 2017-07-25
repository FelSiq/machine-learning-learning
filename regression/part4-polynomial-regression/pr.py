# Get the dataset
import pandas as pd
import numpy as np

dataset = pd.read_csv('/home/felipe/Documentos/Machine Learning A-Z/' + 
	'Part 2 - Regression/Section 6 - Polynomial Regression/Position_Salaries.csv')
ind_param = dataset.iloc[:, :-1].values
dep_param = dataset.iloc[:,  -1].values

# No missing values.
# Hierarchical parameter found.

# Hypothesis: both column #0 and #1 of the dataset has exactly
# the same prediction power, and means exactly the same thing.
# In fact, the #1 column is the #0 codified.
ind_param = np.asmatrix(ind_param[:, -1]).T

# No need to transform the hierarchical parameter to dummy variables,
# as it's values can be expressed with a higher-lower sequence:
# x0 < x1 < x2 < ... < xn

from sklearn.preprocessing import PolynomialFeatures as sklp_pf
from sklearn.linear_model import LinearRegression as skllm_lr

# Plot a polynomial prediction curve ========
def plot_polcurve(ind_param, dep_param, color = 'red', degree = 2, show = False):
	"""
	Crucial explanations:
		- poly_transform = transform the dataset to the polynomial shape of the same dataset
		- new_dataset = dataset of the independent variables with n polynomial x's
		- poly_regressor = linear polynomial model to the "degree", fitted to the n degree
	"""
	# Build the Polynomial Linear Regression Model
	poly_transform = sklp_pf(degree = degree)
	new_dataset = poly_transform.fit_transform(ind_param)
	poly_regressor = skllm_lr()
	poly_regressor.fit(new_dataset, dep_param)
	# High definition plotting
	ind_param_grid = np.asmatrix(np.arange(min(ind_param)[0, 0], max(ind_param)[0, 0] + 0.1, 0.1)).T
	plt.plot(ind_param_grid, poly_regressor.predict(poly_transform.fit_transform(ind_param_grid)), color = color)
	# Show plot, if asked
	if (show):
		plt.show()
	# Return the regressor model
	return (poly_regressor)

# Results ===================================
import matplotlib.pyplot as plt
# Visualise the prediction results
plt.scatter(np.asarray(ind_param), dep_param, color = 'blue')

model_list = []
model_list.append(plot_polcurve(ind_param, dep_param, 'red', 1)) # Linear = Univariate Regression
model_list.append(plot_polcurve(ind_param, dep_param, 'cyan', 2)) # Better, but not good
model_list.append(plot_polcurve(ind_param, dep_param, 'magenta', 4)) # Good!
model_list.append(plot_polcurve(ind_param, dep_param, 'green', 8)) # Excellent?! Beware overfitting!

plt.title('Univariate vs Polynomial Linear Regression')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

# Predict new values
for i in range(0, 4):
	print model_list[i].predict(sklp_pf(degree = pow(2, i)).fit_transform(6.5))