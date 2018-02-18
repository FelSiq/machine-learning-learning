import numpy as np
import math

"""
This is an early version of adaboost. It does not induce classifiers by itself,
it just takes the misclassified instances of each one and return the vote power
of each classifier on the final strong classifier.
"""

class adaboost:
	def _calcAlpha(self, error, ep=1.0e-07):
		return 0.5 * math.log(ep + (1.0 - error) / (error + ep))

	def _updateWeight(self, weight, error, misclassified, ep=1.0e-7):
		return weight * 0.5/(ep + (error if misclassified else (1.0 - error)))

	def emsemble(self, instNum, misclassified, maxIteration=10, verbose=False):
		weights = np.array([1.0/instNum] * instNum)

		alphas = {classifier : 0.0 for classifier in misclassified}

		it = 0
		minError = math.inf
		while it < maxIteration:
			it += 1

			minError = math.inf
			bestClassifier = None

			for classifier in misclassified:
				curError = sum(weights[misclassified[classifier]])

				if curError < minError:
					minError = curError
					bestClassifier = classifier

			if verbose:
				print('Round', it, '\tchoosen classifier:', bestClassifier, '\terror:', minError)
			alphas[bestClassifier] += self._calcAlpha(minError)

			# Update weights
			for n in range(instNum):
				weights[n] = self._updateWeight(weights[n], minError, n in misclassified[bestClassifier])

		return alphas


if __name__ == '__main__':
	instNum = 10
	misclassified = {'A': [1,5,6], 'B': [3,4], 'C': [0,4,6], 'E': [2,3,6]}
	# misclassified = {'B': [0,4,5,8], 'D': [0,3,6,7], 'F': [1,6], 'I': [1,2,7,9]}
	strongerClassifier = adaboost().emsemble(instNum, misclassified, verbose=True, maxIteration=10)
	print(strongerClassifier)
