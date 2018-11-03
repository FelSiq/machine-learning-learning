from math import log
from numpy import array

class LossFunction:
	def surprisal(probs):
		"""
			Note: by logarithm property,
			-log(x) = log(1.0 / x), x > 0
		"""
		if type(probs) is float or type(probs) is int:
			return -log(probs, 2)

		if sum(probs) != 1.0:
			raise ValueError("Probs must sum up to 1.0")

		surprisal_vec = array(\
			[-log(p_val, 2) 
			for p_val in probs])

		return surprisal_vec

	def entropy(probs):
		"""
			Shannon entropy is the weighted
			average of surprisal of each outcome.
			The weights are the probability of
			each outcome to happen.
		"""
		if sum(probs) != 1.0:
			raise ValueError("Sum of probs must be 1.0")

		return sum(array(probs) * LossFunction.surprisal(probs))

	def crossEntropy(true_probs, pred_probs):
		"""
			CrossEntropy is nothing more than
			Shannon's Entropy concepted, but
			the "surprisal" factor is calculated
			using the predicted probabilities
			instead of the true probability.
		"""
		if sum(true_probs) != 1.0 or sum(pred_probs) != 1.0:
			raise ValueError("Sum of probability vectors must be 1.0")

		return sum(true_probs * LossFunction.surprisal(pred_probs))

	def binCrossEntropy(class_probs):
		if len(class_probs) != 2:
			raise ValueError("Length of class probs must be 2.")
		prob_a, prob_b = class_probs

		if prob_a + prob_b != 1.0:
			raise ValueError("Probability must sum 1.0")

		return prob_a * LossFunction.surprisal(prob_a) + \
			(1.0 - prob_a) * LossFunction.surprisal(1.0 - prob_a)

if __name__ == "__main__":
	from collections import OrderedDict
	import sys

	if len(sys.argv) < 2:
		print("usage:", sys.argv[0], 
			"<vals_sep_by_spaces> [sec_vals_sep_by_spaces]")
		exit(1)

	vals_a = list(map(float, sys.argv[1].split(" ")))
	vals_b = None
	if len(sys.argv) >= 3:
		vals_b = list(map(float, sys.argv[2].split(" ")))

	func_args = OrderedDict([
		("surprisal" , (vals_a,)),
		("entropy" , (vals_a,))
	])

	func_addr = [
		LossFunction.surprisal, 
		LossFunction.entropy, 
	]

	if vals_b is not None:
		func_args["cross_entropy"] = (vals_a, vals_b)
		func_addr.append(LossFunction.crossEntropy)

	if len(vals_a) == 2:
		func_args["binary_cross_entropy"] = (vals_a,)
		func_addr.append(LossFunction.binCrossEntropy)
	
	print("Measures calculated:")
	for func_label, func_addr in zip(func_args.keys(), func_addr):
		cur_args = func_args[func_label]
		res = func_addr(*cur_args)
		print(func_label, "\t:", res)

