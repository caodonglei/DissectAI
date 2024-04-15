""" interface of basic neural networks
"""

class NotImpelementError(Exception):
	""" must be implemented by subclass """

class Layer:
	def _forward(self):
		raise NotImpelementError("forward is not implemented.")

	def _backward(self):
		raise NotImpelementError('backward is not implemented.')

	def init_params(self, **kwargs):
		""" initialize network parameter """
		raise NotImpelementError('init_params is not implemented.')

	def train(self, X, Y):
		""" train the model """
		raise NotImpelementError('train is not implemented.')

	def predict(self, X):
		""" predict on the input X """
		raise NotImpelementError('predict is not implemented.')