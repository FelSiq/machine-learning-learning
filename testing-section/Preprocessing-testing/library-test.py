"""
In python, separating the independent variables x, y, z... of the
dependent variable f(x, y, z, ...), with these packages, is a must.

================================================
Step 0: Care with the missing values of the data
Approaches:
	a) Remove lines with the missing data
	b) Fill with the mean/median/mode of the values
"""

# Import section
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Let's fire up!
dataset = pd.read_csv("/home/felipe/Documentos/Machine Learning A-Z/Part 1 - Data Preprocessing/Data.csv")

# Create a independent variable matrix
ind_features = dataset.iloc[:, :-1].values

# Get the output label into a vector 
output_label = dataset.iloc[:, -1].values

# Fill the missing values with the median values =======================================
from sklearn.preprocessing import Imputer # (1)
imputer = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imputer = imputer.fit(ind_features[:, 1:3])
ind_features[:, 1:3] = imputer.transform(ind_features[:, 1:3])

# Discretize categorical values =======================================
from sklearn.preprocessing import LabelEncoder as lbe, OneHotEncoder as ode
# Input column
labencoder_country = lbe()
ind_features[:, 0] = labencoder_country.fit_transform(ind_features[:, 0])

hotencoder = ode(categorical_features = [0])
ind_features = hotencoder.fit_transform(ind_features).toarray()

# Output column
labencoder_output = lbe()
output_label = labencoder_output.fit_transform(output_label)

# Split the data into training set and test set =======================================
from sklearn.cross_validation import train_test_split as tts
ind_train_set, ind_test_set, out_train_set, out_test_set = tts(ind_features, output_label, test_size = 0.3)

# Put all the real parameters with the same scale
# The dummy variable scaling opens a great discussion.
# Largely depends on the context. This time, they will be scaled.
"""
Normalization:
	new_value = (old_value - min)/(max - min)
Standartization:
	new_value = (old_value - mean)/(standard_devitation)
"""
from sklearn.preprocessing import StandardScaler as ss
datascaler = ss()
ind_train_set = datascaler.fit_transform(ind_train_set)
ind_test_set = datascaler.transform(ind_test_set)