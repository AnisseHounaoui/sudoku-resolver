from dnn_framework import Optimizer


class SgdOptimizer(Optimizer):

    def __init__(self, parameters, learning_rate=0.01):
        super().__init__(parameters)
        self._learning_rate = learning_rate

    def _step_parameter(self, parameter, parameter_grad, parameter_name):
        """
        This method returns the new value of the parameter.
        :param parameter: The parameter tensor
        :param parameter_grad: The gradient with respect to the parameter
        :param parameter_name: The parameter name
        :return: The new value of the parameter
        """
        return parameter - self._learning_rate * parameter_grad
