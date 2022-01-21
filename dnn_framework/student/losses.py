from dnn_framework import Loss
import numpy as np


class CrossEntropyLoss(Loss):

    def calculate(self, x, target):
        """
        :param x: The input tensor
        :param target: The target tensor
        :return A tuple containing the loss and the gradient with respect to the input (loss, input_grad)
        """
        target_onehot = np.zeros((target.size, x.shape[1]))
        target_onehot[np.arange(target.size), target] = 1
        loss = - np.sum(np.multiply(target_onehot, np.log(softmax(x))))

        grad = softmax(x) - target_onehot
        return loss, grad


class MeanSquaredErrorLoss(Loss):

    def calculate(self, x, target):
        """
        :param x: The input tensor
        :param target: The target tensor
        :return A tuple containing the loss and the gradient with respect to the input (loss, input_grad)
        """
        loss = np.mean((x - target)**2, keepdims=True)
        dy = 2 * (x - target) / x.size
        loss_cache = (loss, dy)
        return loss_cache


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
