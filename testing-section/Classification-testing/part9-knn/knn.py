# Get the dataset and split into dependent array and independent matrix
import pandas as pd
dataset = pd.read_csv('/home/felipe/Documentos/Machine Learning A-Z/' + 
	'Part 3 - Classification/Section 14 - Logistic Regression/Social_Network_Ads.csv')
ind_var = dataset.iloc[:, :-1].values
dep_var = dataset.iloc[:,  -1].values

# Search for missing values
# No missing data.

# Check for categorical/hierarchical parameters
# Column #0 is useless and #1 is categorical.
from sklearn.preprocessing import LabelEncoder as skl_le 
ind_var = ind_var[:, 1:(ind_var.shape[1] + 1)]
ind_var[:, 0] = skl_le().fit_transform(ind_var[:, 0])

# Discussion: are the 'Gender' parameter relevant? How to evaluate this correctly?
ind_var = ind_var[:, 1:(ind_var.shape[1] + 1)]

# Feature scaling?
from sklearn.preprocessing import StandardScaler as skl_ss
ind_var = skl_ss().fit_transform(ind_var.astype(float))

# Split the dataset into test and train sets
from sklearn.model_selection import train_test_split as skl_tts
ind_train_set, ind_test_set, dep_train_set, dep_test_set = skl_tts(ind_var, dep_var, test_size = 0.25)

# Fit the model
from sklearn.neighbors import KNeighborsClassifier as skl_knc
# p = 2 -> euclidean, p = 1 -> manhatann
classifier = skl_knc(n_neighbors = 5, metric = 'minkowski', p = 2).fit(ind_train_set, dep_train_set)

# Predict the test set
prediction = classifier.predict(ind_test_set)

# Create confusion matrix to verify the results
from sklearn.metrics import confusion_matrix as skl_cm
conf_matrix = skl_cm(y_true = dep_test_set, y_pred = prediction)
conf_matrix