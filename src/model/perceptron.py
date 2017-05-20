# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np

from util.activation_functions import Activation
from model.classifier import Classifier

# from functools import partial

# from sklearn.decomposition import PCA

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
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

        # first we normalize input vectors through negating class 2
        # vectors (non 7 features)
        for i in range(0, len(self.trainingSet.input)):
            if not self.trainingSet.label[i]:
                self.trainingSet.input[i] = np.negative(self.trainingSet.input[i])
        print ("...flipped non 7 feature vectors...")

        # add threshold weight
        self.weight = - np.append(self.weight, np.random.rand(1))/100

        # then start learning the discriminat weights
        print("...updating weights:")
        self.updateWeights(self.trainingSet.input, None, verbose=True)

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
        classification_result = self.fire(np.append(testInstance, [1]))
        if classification_result > 0:
            return True
        else:
            return False

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

    def updateWeights(self, input, error, verbose=False):
        # Write your code to update the weights of the perceptron here
        epoch = 0
        while epoch < self.epochs:
            epoch += 1
            # the vector to hold - J_p(W), the inverse gradient
            gradient = np.zeros(self.weight.shape)
            n_miss = 0
            for instance in self.trainingSet.input:
                # dummy input 1 for threshold weight learning
                if self.fire(np.append(instance, [1])) == 0:
                    n_miss += 1
                    gradient = np.add(
                        gradient, np.append(instance, [1]))

            if not gradient.any():  # class discriminat line found -> stop
                print("epoch " + str(epoch) + " validation accuracy:" +
                      str(1 - np.divide(n_miss, float(len(input)))))
                break
            else:
                if verbose:
                    print("epoch " + str(epoch) + " validation accuracy:" +
                          str(1 - np.divide(n_miss, float(len(input)))))
                    # print(str(n_miss))
                self.weight = np.add(self.learningRate * gradient, self.weight)
                # self.weight = np.add(self.learningRate * gradient, self.weight)

    def fire(self, input):
        """Fire the output of the perceptron corresponding to the input """
        return Activation.sign(np.dot(np.array(input), self.weight))

