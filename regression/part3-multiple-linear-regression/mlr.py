"""
Simple linear regression: 	y = b0 + b1x1
Multiple Linear Regression:	y = b0 + b1x1 + b2x2 + ... + bnxn
"""

# Section one: import the libraries
import pandas as pd # Data handling

# Section two: Get the dataset
dataset = pd.read_csv('/home/felipe/Documentos/Machine Learning A-Z/Part 2' + 
	' - Regression/Section 5 - Multiple Linear Regression/50_Startups.csv')
ind_param = dataset.iloc[:, :-1].values
dep_param = dataset.iloc[:,  -1].values

# Section three: No missing values.

# Section four: Found categorical values, let's discretize it
from sklearn.preprocessing import LabelEncoder as sklpp_le, OneHotEncoder as sklpp_ohe
# Explanations:
#	sklpp_le 	-> discretize values
#	sklpp_ohe 	-> create dummy parameters

# Create simple encoder, calling it's construtor method
enc_discretize = sklpp_le()
# Discretize the State column of the dependent parameters subdataset
ind_param[:, 3] = enc_discretize.fit_transform(ind_param[:, 3])
# Now, it's time to create dummy paramaters of the previous decodifycation.
# Call OneHotEncoder constructor method
enc_distodummy = sklpp_ohe(categorical_features = [3])
ind_param = enc_distodummy.fit_transform(ind_param).toarray()

# Remember the Dummy Variable Trap! (Where redundant linear dependency was created on some linear formulae)
# Note: Of course Python MLR take care of this for us, but, anyway, just to
# be sure.
ind_param = ind_param[:, 1:]

# Not sure about feature scalling.
# Split the dataset into the train and test set
from sklearn.model_selection import train_test_split as sklpp_tts
ind_param_train, ind_param_test, dep_param_train, dep_param_test = sklpp_tts(ind_param, dep_param, test_size = 0.2)

# Now the fun starts.
# Build the multivariable linear regression.
from sklearn.linear_model import LinearRegression as skllm_lr
regressor = skllm_lr()
regressor.fit(ind_param_train, dep_param_train)

# Make predictions
predictions = regressor.predict(ind_param_test)

# No ploting (Because too much dimensions to visualise).

# Is this the optimum model?
# Let's use the backward feature selection method to check it out.
import statsmodels.formula.api as smfa
import numpy as np

# Create the linear coefficient column (because statsmodel does not treat this to us)
# The trick is to create a '1' column to the train dataset:
# y = b0*1 + b1x1 + b2x2 + ... + bnxn
ind_param = np.append(
	arr = np.ones(shape = (50, 1)).astype(int), 
	values = ind_param, 
	axis = 1)

# Ordinary Least Squares = OLS
# Construct the optimum independent variable dataset with the 
# Backward Elemination Feature Selection method
def backward_elim(ind_params, dep_params, sig_level):
	# Copy all parameters into the ideal parameter array
	ind_param_optimum = ind_param
	# Init the Current Pvalue at the maximum possible value
	cur_max_pval = 1.0
	# Loops the backward elimination iterations
	while cur_max_pval > sig_level:
		# Fit the MLR model into the current optimum parameter array
		regressor_OLS = smfa.OLS(endog = dep_param, exog = ind_param_optimum).fit()
		# Get the current maximum PValue of the current optimum parameter arrays
		cur_max_pval = max(regressor_OLS.pvalues)
		print 'Current Max Pvalue: ' + str(cur_max_pval),
		# Check if the current maximum PValue corresponds to a parameter that should be removed on this iteration
		if (cur_max_pval > sig_level):
			# It is a parameter that should be removed by the backward elim. algorithm
			# Get the index of this param.
			cur_maxpval_ind = regressor_OLS.pvalues.tolist().index(cur_max_pval)
			print '\t- Removing \'' + str(cur_maxpval_ind) + '\' index from optimum parameter matrix'
			# remove it form the ideal parameter array
			ind_param_optimum = ind_param_optimum[:, np.arange(ind_param_optimum.shape[1]) != cur_maxpval_ind]
	# End of backward elim. algorithm loop, emit a message
	print '\t- End of the backward elimination.'
	# Return the ideal optimum parameter set
	return ind_param_optimum

ind_param_optimum = backward_elim(ind_param, dep_param, 0.05)

# Manual process:
"""
ind_param_man_opt = ind_param[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = smfa.OLS(endog = dep_param, exog = ind_param_man_opt).fit()
regressor_OLS.pvalues
ind_param_man_opt = ind_param[:, [0, 1, 3, 4, 5]]
regressor_OLS = smfa.OLS(endog = dep_param, exog = ind_param_man_opt).fit()
regressor_OLS.pvalues
ind_param_man_opt = ind_param[:, [0, 3, 4, 5]]
regressor_OLS = smfa.OLS(endog = dep_param, exog = ind_param_man_opt).fit()
regressor_OLS.pvalues
ind_param_man_opt = ind_param[:, [0, 3, 5]]
regressor_OLS = smfa.OLS(endog = dep_param, exog = ind_param_man_opt).fit()
regressor_OLS.pvalues
ind_param_man_opt = ind_param[:, [0, 3]]
regressor_OLS = smfa.OLS(endog = dep_param, exog = ind_param_man_opt).fit()
regressor_OLS.pvalues
"""

# In the end, we end up with a simple linear regression 
# (only R&D Column remained from the backward elim. algorithm)!
# Create test + train sets 
ind_param_train, ind_param_test, dep_param_train, dep_param_test = sklpp_tts(ind_param_optimum, dep_param, test_size = 0.2)
# Fit the model
regressor = skllm_lr()
regressor.fit(ind_param_train, dep_param_train)
# Make predictions
predictions = regressor.predict(ind_param_test)
# Plot the result
import matplotlib.pyplot as plt
plt.scatter(ind_param_train[:, 1], dep_param_train, color = 'red', label = 'Train Set')
plt.scatter(ind_param_test[:, 1], dep_param_test, color = 'green', label = 'Test Set')
plt.plot(ind_param_train[:, 1], regressor.predict(ind_param_train), color = 'blue')
plt.title('Salary vs Experience')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()

# Compare the prediction vs the true value
max_val = max(max(predictions), max(dep_param_test))
plt.scatter(predictions, dep_param_test, color = 'red', label = 'Prediction vs TrueValue')
plt.plot([0, max_val], [0, max_val], color = 'blue', ls = "--", c = 0.3)
plt.title('Salary vs TrueValue')
plt.xlabel('Prediction')
plt.ylabel('TrueValue')
plt.show()