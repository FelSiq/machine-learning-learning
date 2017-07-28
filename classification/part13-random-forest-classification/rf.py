# Get the dataset
import pandas as pd # Data manipulation package
dataset = pd.read_csv('/home/felipe/Documentos/Machine Learning A-Z/Part 3 - Classification/' + 
	'Section 20 - Random Forest Classification/Random_Forest_Classification/Social_Network_Ads.csv')

# Check for missing values
# No missing values

# Check for categotical/hierarchical parameters
# No categorical neither hierarchical parameters, but the first two
# features are futile for this model.
dataset = dataset.iloc[:, 2:dataset.shape[1]]

# Split the dataset into independent matrix and 
# dependent array (python's way of machine learning)
ind_param = dataset.iloc[:, :-1].values
dep_param = dataset.iloc[:,  -1].values

# Feature scaling
# Random forest is not a distance-based algorithm, so it does not
# demand feature scaling.
from sklearn.preprocessing import StandardScaler as skl_ss
ind_param = skl_ss().fit_transform(ind_param.astype(float))

# Split the dataset into train and test sets
from sklearn.model_selection import train_test_split as skl_tts
ind_train_set, ind_test_set, dep_train_set, dep_test_set = skl_tts(ind_param, dep_param, test_size = 0.25)

# Fit the model
from sklearn.ensemble import RandomForestClassifier as skl_rfc
classifier = skl_rfc(
	n_estimators = 1000,
	criterion = 'entropy')
classifier.fit(ind_train_set, dep_train_set)

# Predict the test set
predictions = classifier.predict(ind_test_set)

# Confusion matrix
from sklearn.metrics import confusion_matrix as skl_cm
skl_cm(y_true = dep_test_set, y_pred = predictions)

# Plot the results
# =============================================
# Disclaimer: All the plotting code below is not mine.
# Font: Machine Learning A-Z™: Hands-On Python & R In Data Science - kernel_svm.py source code 
# =============================================
# Caution: Don't run that with a large (e.g 1000) number of trees/estimators without a powerful machine.
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt 
import numpy as np

X_set, y_set = ind_train_set, dep_train_set

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
	plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)

plt.title('Kernel SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
# ============================================