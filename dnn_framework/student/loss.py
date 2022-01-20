import numpy as np

from dnn_framework.loss import Loss

##########################################  CrossEntropyLoss function ########################################

class CrossEntropyLoss(Loss):

    """cross entropy loss function"""

    def calculate(self, x, target):
        """
        :param x: The input tensor
        :param target: The target tensor
        :return A tuple containing the loss and the gradient with respect to the input (loss, input_grad)
        """
        loss = -np.sum(np.multiply(target, np.log(x)))
        dy = -(target / x)
        loss_cache = (loss, dy)
        return loss_cache
      
        raise NotImplementedError()
        
   ##########################################  MeanSquareErrorLoss function ########################################
    
class MeanSquareErrorLoss(Loss):

    """Mean square error loss function"""

    def calculate(self, x, target):
        """
        :param x: The input tensor
        :param target: The target tensor
        :return A tuple containing the loss and the gradient with respect to the input (loss, input_grad)
        """
        loss = np.sum(x - target, keepdims=True)
        dy = 2 * (x - target)
        loss_cache = (loss, dy)
        return loss_cache
        raise NotImplementedError()
