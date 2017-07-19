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

# Build a Univariate Linear Regression Model (just as basis of comparison)
# Build the Polynomial Linear Regression Model
from sklearn.linear_model import LinearRegression as skllm_lr
lin_reg = skllm_lr()
lin_reg.fit(ind_param, dep_param)

# Compare the two results
from sklearn.preprocessing import PolynomialFeatures as sklp_pf
ind_param_poly_degree2 = sklp_pf(degree = 2).fit_transform(ind_param)
ind_param_poly_degree4 = sklp_pf(degree = 4).fit_transform(ind_param)
ind_param_poly_degree8 = sklp_pf(degree = 8).fit_transform(ind_param)
# How to define the correct degree? This must be verified! [x] 
pol_reg_degree2 = skllm_lr()
pol_reg_degree4 = skllm_lr()
pol_reg_degree8 = skllm_lr()
pol_reg_degree2.fit(ind_param_poly_degree2, dep_param)
pol_reg_degree4.fit(ind_param_poly_degree4, dep_param)
pol_reg_degree8.fit(ind_param_poly_degree8, dep_param)

# Results ===================================
import matplotlib.pyplot as plt
# 	Visualise the univariate linear regressor results
# + Visualise the polynomial linear regressor reuslts
plt.scatter(np.asarray(ind_param), dep_param, color = 'blue')
plt.plot(ind_param, lin_reg.predict(ind_param), color = 'red')
plt.plot(ind_param, pol_reg_degree2.predict(ind_param_poly_degree2), color = 'green')
plt.plot(ind_param, pol_reg_degree4.predict(ind_param_poly_degree4), color = 'gray')
plt.plot(ind_param, pol_reg_degree8.predict(ind_param_poly_degree8), color = 'black')
plt.title('Univariate vs Polynomial Linear Regression')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()