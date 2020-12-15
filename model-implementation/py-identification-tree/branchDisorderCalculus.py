import numpy as np
import math


def subsetEntropy(subset, base=2):
    classes, absFreqs = np.unique(subset, return_counts=True)
    probs = absFreqs / len(subset)
    return -sum(
        [(probs[i] * math.log(probs[i], base) if 0.0 < probs[i] < 1.0 else 0.0)
         for i in range(len(probs))])


def setEntropy(instSet, base=2):
    if len(instSet):
        totalSetLen = 0
        for subset in instSet:
            totalSetLen += len(subset)
        totalDisorder = 0.0
        for subset in instSet:
            if len(subset):
                totalDisorder += subsetEntropy(
                    subset, base) * (len(subset) / totalSetLen)
        return totalDisorder
    return math.inf


# WC >= 450
# print(setEntropy([[1,1,1,1,0,0], [0,0,0,0,0,0]]))
# WC >= 350
# print(setEntropy([[1,1,1,0], [1,0,0,0,0,0,0,0]]))
"""
instances = [(30,0),(30, 0), (25, 0), (20,0), (20, 1), (15, 0), (10, 0), (10, 1), (7.5, 0), (5, 1), (2.5, 1), (0, 1)]

step = 31/10000
threshold = [i * step for i in range(0, 10000)]

bestVal = math.inf
bestThreshold = 0.0
for t in threshold:
	groupA = []
	groupB = []
	for n in instances:
		if n[0] >= t:
			groupA.append(n[1])
		else:
			groupB.append(n[1])
	aux = setEntropy([groupA, groupB])
	if aux < bestVal:
		bestVal = aux
		bestThreshold = t


print('Y >=', bestThreshold, '\tEntropy:', bestVal)
"""
