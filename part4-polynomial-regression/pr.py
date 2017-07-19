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
def plot_polcurve(ind_param, dep_param, color = 'red', degree = 2):
	"""
	Crucial explanations:
		- new_dataset = dataset of the independent variables with n polynomial x's
		- poly_regressor = linear polynomial model to the "degree", fitted to the n degree
	"""
	# Build the Polynomial Linear Regression Model
	new_dataset = sklp_pf(degree = degree).fit_transform(ind_param)
	poly_regressor = skllm_lr()
	poly_regressor.fit(new_dataset, dep_param)
	plt.plot(ind_param, poly_regressor.predict(new_dataset), color = color)

# Results ===================================
import matplotlib.pyplot as plt
# 	Visualise the univariate linear regressor results
# + Visualise the polynomial linear regressor reuslts
plt.scatter(np.asarray(ind_param), dep_param, color = 'blue')
# plt.plot(ind_param, lin_reg.predict(ind_param), color = 'red')

plot_polcurve(ind_param, dep_param, 'red', 1) # Linear = Univariate Regression
plot_polcurve(ind_param, dep_param, 'green', 2)
plot_polcurve(ind_param, dep_param, 'green', 4)
plot_polcurve(ind_param, dep_param, 'green', 8)

plt.title('Univariate vs Polynomial Linear Regression')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()