import numpy as np
import math

"""
This is a early version of adaboost. It does not tries classifiers by itself already,
it's just a prototype.
"""

class adaboost:
	def _calcAlpha(self, error):
		return 0.5 * math.log((1.0 - error) / error)

	def _updateWeight(self, weight, error, misclassified):
		return weight * 0.5 * 1.0/(error if misclassified else (1.0 - error))

	def emsemble(self, instNum, misclassified, maxIteration=4, halfwayError=True, verbose=False):
		weights = np.array([1.0/instNum] * instNum)

		alphas = {classifier : 0.0 for classifier in misclassified}

		it = 0
		minError = math.inf
		while it < maxIteration and minError != 0.5:
			it += 1

			minError = -math.inf if halfwayError else math.inf
			bestClassifier = None


			for classifier in misclassified:
				curError = sum(weights[misclassified[classifier]])

				if halfwayError:
					curError = abs(curError - 0.5)

				# if (not halfwayError and curError < minError) or (halfwayError and curError >= minError):
				if halfwayError ^ (curError < minError):
					minError = curError
					bestClassifier = classifier

			if not halfwayError or minError != 0.5:
				if verbose:
					print('Round', it, '\tchoosen classifier:', bestClassifier, '\terror:', minError)
				alphas[bestClassifier] += self._calcAlpha(minError)

				# Update weights
				for n in range(instNum):
					weights[n] = self._updateWeight(weights[n], minError, n in misclassified[bestClassifier])
			elif verbose:
				print('Reached min error of 0.5. Stopping boosting.')

		return alphas


if __name__ == '__main__':
	instNum = 8
	misclassified = {'A': [1,5,6], 'B': [3,4], 'C': [0,4,6], 'E': [2,3,6]}
	strongerClassifier = adaboost().emsemble(instNum, misclassified, halfwayError=False, verbose=True)
	print(strongerClassifier)
