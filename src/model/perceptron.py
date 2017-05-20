# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np

from util.activation_functions import Activation
from util.loss_functions import DifferentError
from model.classifier import Classifier

# from functools import partial

# from sklearn.decomposition import PCA

logging.basicConfig(format='[%(levelname)s]: epoch duration '
                           '%(asctime)s.%(msecs)03ds; %(message)s',
                    datefmt="%S",
                    level=logging.DEBUG,
                    stream=sys.stdout)


class Perceptron(Classifier):
    """
    A digit-7 recognizer based on perceptron algorithm

    Parameters
    ----------
    train : list
    valid : list
    test : list
    learningRate : float
    epochs : positive int

    Attributes
    ----------
    learningRate : float
    epochs : int
    trainingSet : list
    validationSet : list
    testSet : list
    weight : list
    """
    def __init__(self, train, valid, test,
                                    learningRate=0.01, epochs=50):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        # print ([x for x in self.trainingSet.input[:10]])

        # Initialize the weight vector with small random values
        # around 0 and0.1
        self.weight = np.random.rand(self.trainingSet.input.shape[1])/100
                    # np.ones(self.trainingSet.input.shape[1])

    def train(self, verbose=True):
        """Train the perceptron with the perceptron learning algorithm.


        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """

        # first we normalize input vectors through negating all feature vectors
        # corresponding to non 7 features (label == False)
        normalized_input = [np.negative(self.trainingSet.input[i])
                            if not self.trainingSet.label[i]
                            else self.trainingSet.input[i]
                            for i in range(0, len(self.trainingSet.input))]

        # no threshhold discriminant estimation
        input = normalized_input
        # threshhold discriminant estimation:
        # self.weight = np.append(self.weight, np.random.rand(1))/100
        # input = map(lambda x: np.append(x, [1]), normalized_input)1

        # start learning the discriminat weights
        error = DifferentError()
        epoch = 0
        while epoch < self.epochs:
            epoch += 1
            n_err = self.updateWeights(input, error)
            if verbose:
                accuracy_string = 'accuracy %(acc)02.02f'\
                                  % {'acc': 100 * (1.0 - float(n_err) / len(input))}
                logging.debug("(epoch %(epoch)02d) %(acc)s%%"
                              %{'epoch':epoch, 'acc': accuracy_string})
            if n_err == 0:
                return

    def classify(self, testInstance):
        """Classify a single instance.

        Parameters
        ----------
        testInstance : list of floats

        Returns
        -------
        bool :
            True if the testInstance is recognized as a 7, False otherwise.
        """
        return self.fire(testInstance)

    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.testSet.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))

    def updateWeights(self, input, error):
            n_miss = 0
            i = 0
            error_gradient = np.zeros(len(input[0]))
            while i < len(input):
                instance = input[i]
                err = error.calculateError(1, self.fire(instance))
                if err:
                    n_miss += 1
                    error_gradient += instance
                i += 1
            if not error_gradient.any():
                return n_miss
            else:
                self.weight = np.add(self.learningRate * error_gradient,
                                     self.weight)
            return n_miss

    def fire(self, input):
        """Fire the output of the perceptron corresponding to the input """
        return Activation.sign(np.dot(np.array(input), self.weight))

