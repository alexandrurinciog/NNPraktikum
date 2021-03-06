# -*- coding: utf-8 -*-


"""
Loss functions.
"""

import numpy as np

from abc import ABCMeta, abstractmethod, abstractproperty


class Error:
    """
    Abstract class of an Error
    """
    __metaclass__ = ABCMeta

    @abstractproperty
    def errorString(self):
        pass

    @abstractmethod
    def calculateError(self, target, output):
        # calculate the error between target and output
        pass

    @abstractmethod
    def calculateErrorPrime(self, target, output):
        # calculate output dependent error gradient
        pass


class AbsoluteError(Error):
    """
    The Loss calculated by the number of differences between target and output
    """
    def errorString(self):
        self.errorString = 'absolute'

    def calculateError(self, target, output):
        # It is the numbers of differences between target and output
        return abs(target - output)


class DifferentError(Error):
    """
    The Loss calculated by the number of differences between target and output
    """
    def errorString(self):
        self.errorString = 'different'

    def calculateError(self, target, output):
        # It is the numbers of differences between target and output
        return target - output


class MeanSquaredError(Error):
    """
    The Loss calculated by the mean of the total squares of differences between
    target and output.
    """
    def errorString(self):
        self.errorString = 'mse'

    def calculateError(self, target, output):
        # Here you have to code the MSE
        # MSE = 1/n*sum (i=1 to n) of (target_i - output_i)^2)
        mse = 1.0 / float(len(target)) * np.sum([pow(target[i] - output[i], 2)
                            for i in range(0, target.shape[0])])
        return mse

    def calculateErrorPrime(self, target, output):
        # calculate output dependent error gradient forall samples in output
        n = target.shape[0]
        return map(lambda i: 2.0 / float(n) * (target[i] - output[i]), range(0, n))


class SumSquaredError(Error):
    """
    The Loss calculated by the sum of the total squares of differences between
    target and output.
    """
    def errorString(self):
        self.errorString = 'sse'

    def calculateError(self, target, output):
        # Here you have to code the SSE
        # SSE = 1/2*sum (i=1 to n) of (target_i - output_i)^2)
        sse = 0.5 * np.sum([pow(target[i] - output[i], 2)
                                        for i in range(0, target.shape[0])])
        return sse

    def calculateErrorPrime(self, target, output):
        # calculate output dependent error gradient forall samples in output
        return map(lambda i: target[i] - output[i], range(0, target.shape[0]))

class BinaryCrossEntropyError(Error):
    """
    The Loss calculated by the Cross Entropy between binary target and
    probabilistic output (BCE)
    """
    def errorString(self):
        self.errorString = 'bce'

    def calculateError(self, target, output):
        pass


class CrossEntropyError(Error):
    """
    The Loss calculated by the more general Cross Entropy between two
    probabilistic distributions.
    """
    def errorString(self):
        self.errorString = 'crossentropy'

    def calculateError(self, target, output):
        pass
