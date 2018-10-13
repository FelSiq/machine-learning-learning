from numpy import array, zeros, diag, random, \
	where, concatenate, delete, c_ as cat_cols, \
	r_ as cat_rows, nan
from pandas import DataFrame
import matplotlib.pyplot as plt

class Valclass():
	def generalization():
		# Generalization = |R(f) - R_emp(f)|
		# R(f) : "actual risk" or "expected value"
		# R_emp(f) : "empirical risk"
		pass

	def empirical_risk(confusion_matrix):
		# Risk_emp(f) = 1.0 - Accuracy(f) 
		return 1.0 - Valclass.accuracy(confusion_matrix)

	def confusion_matrix(true_labels, class_result):
		"""
		Confusion Matrix:
		o-------o-------o-------o-------o
		|||||||||T_Pos	|T_Neg	|||||||||
		o-------o-------o-------o-------o
		|P_Pos	|TPos	|FNeg	|Sum_D	|
		o-------o-------o-------o-------o
		|P_Neg	|FPos	|TNeg	|Sum_C	|
		o-------o-------o-------o-------o
		|||||||||Sum_A	|Sum_B	|Sum_T	|
		o-------o-------o-------o-------o
		"""

		class_labels = set(true_labels)

		if set(class_result) - class_labels:
			print("Error: more classes predicted than",
				"specified in the true label array:",
				set(class_result) - class_labels)
			return None

		if type(true_labels) != type(array):
			true_labels = array(true_labels)

		if type(class_result) != type(array):
			class_result = array(class_result)

		class_labels = list(class_labels)

		num_classes = len(class_labels)

		m = array([
			[sum(class_result[true_labels \
				== class_labels[cl_id_1]] \
					== class_labels[cl_id_2])
			for cl_id_2 in range(num_classes)]
			for cl_id_1 in range(num_classes)
		])

		m = DataFrame(m, 
			index=class_labels, 
			columns=class_labels, 
			dtype=float)

		return m

	def accuracy(confusion_matrix):
		# (TPos + TNeg) / (TPos + FNeg + FPos + TNeg)
		#	== (TPos + TNeg) / Sum_T
		return sum(diag(confusion_matrix)) /\
				confusion_matrix.values.sum()

	def precision(confusion_matrix, true_class_index=0):
		# TPos / (TPos + FPos) == TPos / Sum_A

		return confusion_matrix.values[true_class_index, \
			true_class_index] / (confusion_matrix.iloc[:, \
				true_class_index].values.sum())

	def recall(confusion_matrix, true_class_index=0):
		# True Positive Rate == Recall == Sensitivity
		# TPos / (TPos + FNeg) == TPos / Sum_D

		return confusion_matrix.values[true_class_index, \
			true_class_index] / (confusion_matrix.iloc[\
				true_class_index,:].values.sum())

	def sensitivity(confusion_matrix, true_class_index=0):
		# True Positive Rate == Recall == Sensitivity
		# TPos / (TPos + FNeg) == TPos / Sum_D
		return Valclass.recall(confusion_matrix, 
			true_class_index)

	def tpr(confusion_matrix, true_class_index=0):
		# True positive Rate == Recall == Sensitivity
		# TPos / (TPos + FNeg) == TPos / Sum_D
		return Valclass.recall(confusion_matrix, 
			true_class_index)

	def fnr(confusion_matrix, true_class_index=0):
		# False Negative rate
		# FNeg / (FNeg + TPos) 
		#		== FPos / Sum_D
		#		== 1.0 - TPos / (TPos + FNeg)
		#		== 1.0 - TruePositiveRate
		return 1.0 - Valclass.tpr(confusion_matrix,
			true_class_index)

	def specifity(confusion_matrix, true_class_index=0):
		# True Negative Rate == Specifity
		# TNeg / (TNeg + FPos) = TNeg / Sum_C
		negative_ind = [i for i in range(confusion_matrix.shape[0])]
		negative_ind.remove(true_class_index)

		return confusion_matrix.iloc[negative_ind, \
				negative_ind].values.sum() /\
			confusion_matrix.iloc[negative_ind,:].\
				values.sum()

	def fpr(confusion_matrix, true_class_index=0):
		# False Positive rate
		# FPos / (FPos + TNeg) 
		#	== FPos / Sum_C
		#	== 1.0 - TNeg / (TNeg + FPos)
		#	== 1.0 - Specifity

		return 1.0 - Valclass.specifity(\
			confusion_matrix,
			true_class_index)

	def tnr(confusion_matrix, true_class_index=0):
		# True Negative rate == Specifity
		# TNeg / (TNeg + FPos) = TNeg / Sum_C
		return Valclass.specifity(\
			confusion_matrix,
			true_class_index)

	def fbscore(confusion_matrix, b, true_class_index=0):
		# (1 + b) * precision * recall / 
		# 	(b * precision + recall)
		prec = Valclass.precision(\
			confusion_matrix, 
			true_class_index=true_class_index)

		reca = Valclass.recall(\
			confusion_matrix, 
			true_class_index=true_class_index)

		return (1.0 + b) * prec * reca / (b * prec + reca)

	def f1score(confusion_matrix, true_class_index=0):
		# fbscore with b = 1.
		return Valclass.fbscore(\
			confusion_matrix, 
			b=1.0,
			true_class_index=true_class_index)

	def __simpsonrule__(a, b, values):
		total_inst = len(values)
		h = (b - a)/(total_inst-1)
		odd_sum = 4 * sum(values[1:-1:2]) 
		even_sum = 2 * sum(values[2:-1:2])
		totalsum = values[0] + values[-1] +\
			odd_sum + even_sum
		return (h/3.0) * totalsum

	def roc_auc(tpr, fpr, true_class_index=0, plot=False):
		# ROC stands for "Receiver Operating 
		# Characteristics". AUC stands for "Area Under
		# the Curve (ROC)".

		if len(tpr) != len(fpr):
			print("Error: size of TPR and FPR",
				"arrays must match!")
			return None

		tpr = array(tpr).flatten()
		fpr = array(fpr).flatten()

		# In order to use Simpson Rule for numeric
		# integration the number of points must be
		# odd.
		if len(tpr) % 2 == 0:
			tpr = tpr[:-1]
			fpr = fpr[:-1]

		auc_val = Valclass.__simpsonrule__(\
			a=0.0,
			b=1.0,
			values=tpr)

		if plot:
			aux_data = cat_cols[fpr, tpr]
			print(aux_data.shape)
			aux_data = cat_rows[[[0.0, 0.0]], aux_data]
			aux_data = cat_rows[aux_data, [[1.0, 1.0]]]
			aux_data.sort(axis=0)

			fpr = aux_data[:,0]
			tpr = aux_data[:,1]

			# Plotting
			plt.plot(fpr, tpr, "-o")
			plt.plot([0.0, 1.0], [0.0, 1.0], "r-")
			plt.title("ROC AUC")
			plt.show()

		return auc_val

	def testall(confusion_matrix, b=1, true_class_index=0):
		ans = {
			"accuracy" : \
				Valclass.accuracy(confusion_matrix),

			"empirical_risk" : \
				Valclass.empirical_risk(confusion_matrix),

			"precision" : \
				Valclass.precision(confusion_matrix, 
					true_class_index=true_class_index),

			"true_positive_rate/recall/sensitivity" : \
				Valclass.tpr(confusion_matrix, 
					true_class_index=true_class_index),

			"true_negative_rate/specifity" : \
				Valclass.tnr(confusion_matrix, 
					true_class_index=true_class_index),

			"false_positive_rate" : \
				Valclass.fpr(confusion_matrix, 
					true_class_index=true_class_index),

			"false_negative_rate" : \
				Valclass.fnr(confusion_matrix, 
					true_class_index=true_class_index),

			"f"+str(b)+"-score" : \
				Valclass.fbscore(confusion_matrix,
					b=b,
					true_class_index=true_class_index),
		}

		return ans

	def combine_matrices(confusions_matrix, 
		b=1, 
		true_class_index=0, 
		macroaverage=False):
		"""
			This method combine the results of various
			confusion matrices. It can just sum up all
			matrices and then perform the evaluations,
			or it can perform the evaluations one by one
			and then combine the metrics, averaging then
			and also collecting the standard deviation.
		"""
		if macroaverage:
			# Sum up all confusion matrices and
			# then perform the calculations
			master_matrix = confusions_matrix[0]
			for i in range(1, len(confusions_matrix)):
				master_matrix += confusions_matrix[i]

			return testall(master_matrix, 
				b=b, 
				true_class_index=true_class_index)

		else:
			# Perform the calculations of each matrix
			# separately and then combine they, avera-
			# ging the values. Also, collect the stand-
			# ard deviation of each metric.
			final_ans = {}
			for i in range(1, len(confusions_matrix)):
				ans = testall(confusions_matrix[i],
					b=b,
					true_class_index=true_class_index)

				for item in ans:
					if item not in final_ans:
						final_ans[item] = array([0.0] * \
							len(confusions_matrix))
					final_ans[item][i] = ans[item]

			for item in final_ans:
				aux_std = final_ans[item].std()
				final_ans[item] = {
					"average" : final_ans[item].sum() /\
						len(confusions_matrix),
					"stdev" : aux_std
				}
			
			return final_ans

class Partitions():

	def __getclassprobs__(y, stratified=True):

		if type(y) != type(array):
			y = array(y)
		else:
			y = y.flatten()

		num_inst = len(y)

		# Calculate probability of each class to be
		# selected
		class_labels = set(y)
		class_probs = {}
		for cl in class_labels:
			class_probs[cl] = sum(y == cl)/num_inst

		# Probability of each instance to be chosen
		# in each bin
		if stratified:
			inst_sel_prob = array([0.0] * num_inst)

			for inst_id in range(num_inst):
				inst_sel_prob[inst_id] = \
					class_probs[y[inst_id]]

			inst_sel_prob /= sum(inst_sel_prob)
		else:
			inst_sel_prob = None

		return num_inst, class_labels, inst_sel_prob

	def kfold(y, k=10, stratified=True, ret_train_bins=True):
		"""
			This method separates the train set
			into k mutually exclusive partitions.

			If "stratified" is true, each partition
			will try to keep the original proportion 
			of each class. This is not guaranteed.

			If "ret_train_bins" is true, this method
			will return two lists: test_bins and tr-
			ain_bins which keeps, respectivelly, the
			indexes of test instances and train instan-
			ces in for each iteration of the k-fold cv.

			The purpose behind the creation of the
			train_bin (which effectively is 
			set(all_indexes) - set(test_indexes)) is
			just to make the testing phase simpler.
		"""

		num_inst, class_labels, inst_sel_prob =\
			Partitions.__getclassprobs__(y,
				stratified=stratified)

		# Prepare test bins
		test_bins = []

		if ret_train_bins:
			# Train bins are created just to make 
			# life simplier, not because they're
			# really necessary.
			train_bins = []

		for i in range(k):
			cur_partition_indexes = random.choice(\
				a=range(num_inst),
				size=num_inst//k,
				replace=False,
				p=inst_sel_prob)

			if stratified:
				inst_sel_prob[cur_partition_indexes] = 0.0
				aux_sum = sum(inst_sel_prob)
				if aux_sum > 0.0:
					inst_sel_prob /= aux_sum

			test_bins.append(cur_partition_indexes)

			if ret_train_bins:
				train_bins.append(\
					delete(range(num_inst),
						cur_partition_indexes))

		# Check if there is not remaining
		# instances not chosen. In this case,
		# distribute they in a round-robin fashion
		if sum(inst_sel_prob) > 0.0:
			remaining_ids = array(where(inst_sel_prob > 0.0)).flatten()
			for i in range(len(remaining_ids)):
				test_bins[i % k] = concatenate((\
					test_bins[i % k], [remaining_ids[i]]))
				train_bins[i % k] = delete(train_bins[i % k], \
					where(train_bins[i % k] == remaining_ids[i])[0])

		if ret_train_bins:
			return train_bins, test_bins

		return test_bins

	def holdout(y, test_prop=0.25, stratified=True):
		"""
			This function separates the dataset into
			two mutually exclusive partitions, one
			designed to be used for training a machine
			learning model and the other one to evaluate
			it, trying to approximate its "true risk".
		"""
		num_inst, class_labels, inst_sel_prob =\
			Partitions.__getclassprobs__(y,\
				stratified=stratified)

		test_indexes = random.choice(\
			num_inst,
			size=int(test_prop * num_inst),
			replace=False,
			p=inst_sel_prob)

		train_indexes = delete(\
			range(num_inst),
			test_indexes)
		
		return train_indexes, test_indexes

	def bootstrap(y,
		sep_test=False,
		bag_size_prop=1.0,
		test_prop=0.2, 
		stratified=True):

		num_inst, class_labels, inst_sel_prob =\
			Partitions.__getclassprobs__(y, \
				stratified=stratified)

		if sep_test:
			# Separate instances for test
			test_indexes = random.choice(
				num_inst,
				size=int(test_prop * num_inst),
				replace=False,
				p=inst_sel_prob)

			inst_sel_prob[test_indexes] = 0.0
			inst_sel_prob /= sum(inst_sel_prob)

		# Choose, with replacement, in the training
		# data the instances that are not already in the
		# testing set, if any
		train_bag = random.choice(
			num_inst,
			size=int(bag_size_prop * num_inst),
			replace=True,
			p=inst_sel_prob)

		if sep_test:
			return train_bag, test_indexes

		return train_bag

if __name__ == "__main__":
	from sklearn import datasets
	from sklearn.tree import DecisionTreeClassifier as dtc
	iris_data = datasets.load_iris()

	vals=iris_data["data"]
	target=iris_data["target"]

	from numpy import random, c_ as add_col
	shuffled_data = add_col[vals, target]
	random.shuffle(shuffled_data)

	vals = shuffled_data[:,:-1]
	target = shuffled_data[:,-1]

	dt = dtc(criterion="gini", splitter="best")

	tpr = []
	fpr = []
	for i in range(1,vals.shape[0]-1):
		classifier = dt.fit(X=vals[:i,:], y=target[:i])

		preds_array = classifier.predict(vals[i:,:])
		true_labels = target[i:]

		# Class "0" vs all
		cm = Valclass.confusion_matrix(true_labels, preds_array)

		aux1 = Valclass.tpr(cm)
		aux2 = Valclass.fpr(cm)

		if aux1 is not nan and aux2 is not nan:
			tpr.append(aux1)
			fpr.append(aux2)

	auc = Valclass.roc_auc(tpr=tpr, fpr=fpr, plot=True)

	print("AUC:", auc)
