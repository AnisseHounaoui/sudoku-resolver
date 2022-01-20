import numpy as np

from dnn_framework.optimizer import Optimizer

class SgdOptimizer(Optimizer):

    def __init__(self,parameters,learning_rate=0.01):
        #self._parameters = parameters
        self._learning_rate = learning_rate

        raise NotImplementedError()

    def _step_parameter(self, parameter, parameter_grad, parameter_name):
        """
        This method returns the new value of the parameter.
        :param parameter: The parameter tensor
        :param parameter_grad: The gradient with respect to the parameter
        :param parameter_name: The parameter name
        :return: The new value of the parameter
        """
        parameter[parameter_name] = parameter[parameter_name] - self._learning_rate * parameter_grad[parameter_name]
        
        
        raise NotImplementedError()
