from numpy import array, zeros, diag, random, where, concatenate, delete
from pandas import DataFrame

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

		reca = Valclass.precision(\
			confusion_matrix, 
			true_class_index=true_class_index)

		return (1.0 + b) * prec * reca / (b * prec + reca)

	def f1score(confusion_matrix, true_class_index=0):
		# fbscore with b = 1.
		return Valclass.fbscore(\
			confusion_matrix, 
			b=1.0,
			true_class_index=true_class_index)

	def roc(confusion_matrix, true_class_index=0, plot=False):
		# ROC stands for "Receiver Operating 
		# Characteristics"
		pass

	def testall(confusion_matrix, b=1, true_class_index=0):
		ans = {
			"accuracy" : Valclass.accuracy(confusion_matrix),
			"empirical_risk" : Valclass.empirical_risk(confusion_matrix),
			"recall" : Valclass.recall(confusion_matrix, true_class_index=true_class_index),
			"precision" : Valclass.precision(confusion_matrix, true_class_index=true_class_index),
			"sensitivity" : Valclass.sensitivity(confusion_matrix, true_class_index=true_class_index),
			"true_positive_rate" : Valclass.tpr(confusion_matrix, true_class_index=true_class_index),
			"true_negative_rate" : Valclass.tnr(confusion_matrix, true_class_index=true_class_index),
			"false_positive_rate" : Valclass.fpr(confusion_matrix, true_class_index=true_class_index),
			"false_negative_rate" : Valclass.fnr(confusion_matrix, true_class_index=true_class_index),
			"specifity" : Valclass.specifity(confusion_matrix, true_class_index=true_class_index),
			"f"+str(b)+"-score" : Valclass.fbscore(confusion_matrix,b=b,true_class_index=true_class_index),
			"roc" : Valclass.roc(confusion_matrix, true_class_index=0, plot=False),
		}

		return ans

class Partitions():

	def __getclassprobs__(y, stratified=True):
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

	def kfold(x, y, k=10, stratified=True, ret_train_bins=True):
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

	def holdout(dataset, 
		train_prop=0.75, 
		stratified=True):
		"""
			This function separates the dataset into
			two mutually exclusive partitions, one
			designed to be used for training a machine
			learning model and the other one to evaluate
			it, trying to approximate its "true risk".
		"""
		num_inst, class_labels, inst_sel_prob =\
			Partitions.__getclassprobs__(y, \
				stratified=stratified)

		test_indexes = random.choice(\
			dataset.shape[0], 
			replace=False,
			p=inst_sel_prob)

		train_indexes = delete(\
			range(dataset.shape[0]),
			test_indexes)
		
		return train_indexes, test_indexes

	def bootstrap(dataset, 
		train_func, 
		rep=1000, 
		train_prop=0.75,
		macroaverage=True,
		full_output=False):

		# Random sampling with replacement

		"""
			Amostragem com reposição

			Cada partição é uma amostra aleatória com reposição 
			do conjunto total de exemplos

			Conjunto de treinamento têm o mesmo número de exemplos
			do conjunto total

			Esta  reamostragem é feita muitas vezes(de 1000 a 10000 
			vezes) para criar uma estimativa da função de distribuição 
			acumulada.

			Processo é repetido k vezes. Resultado final é a média dos k 
			experimentos.
		"""
		pass

if __name__ == "__main__":
	m = Valclass.confusion_matrix(
		true_labels=([1, 2, 1, 3]), 
		class_result=([1, 2, 1, 1]))

	t_cl_ind = 1

	print(m)

	ans = Valclass.testall(m, true_class_index=t_cl_ind)

	print("\nResults:")
	max_len = 1 + max([len(k) for k in array(list(ans.keys()))])
	for item in ans:
		print("{val:<{fill}}".format(val=item, fill=max_len), ":", ans[item])

	from sklearn.datasets import load_iris
	iris = load_iris()
	k = 10
	train, test = Partitions.kfold(iris["data"], iris["target"], k=k)

	for i in range(k):
		print(set(train[i]).intersection(test[i]))
