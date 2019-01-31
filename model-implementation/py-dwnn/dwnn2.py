class dwnn:
    """
	The larger the 'sigma' is, the wider the
	of the gaussian weight function will be.

	In a mathematic view, this imply that the
	larger the sigma is, the smaller the bias of
	the model will be, so the regression task will
	tend to underfitting (the model will tend to
	always produce the arithmetic average as output)
	On the other side, the smaller the sigma is, the
	larger is the bias of the model, and it will
	tend to produce overfitted results.

	Bias and Sigma have a inverse proportionality:
	Bias	: up	down
	Sigma	: down	up
	"""

    def _euclideanDist(self, a, b):
        return (sum(a**2 - b**2))**0.5

    def _gaussian(self, dist, sigma=1.0):
        return np.exp(-dist**2.0 / (2.0 * sigma**2.0))

    def feed(self, query, x, y, sigma=1.0):

        wSum = 0.0
        for s, val in zip(x, y):
            w = self._gaussian(self._euclideanDist(query, s), sigma)
            ret = w * val
            wSum += w
        return ret / wSum
