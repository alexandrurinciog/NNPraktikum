# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np
import matplotlib.pyplot as plt

import util.loss_functions as erf
from util.activation_functions import Activation
from model.classifier import Classifier



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
        error = erf.DifferentError()
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


class LogisticNeuron(Classifier):
    """
    A digit-7 recognizer based on perceptron algorithm with a mean squared error
    loss function and sigmoid activation function.

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
                 learningRate=0.01, epochs=50,
                 activation='sigmoid',
                 error='mse'):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        # Initialize the weight vector with small random values
        # between -0.3 and 0.3 to encourage sigmoid function learning
        self.weight = np.random.rand(self.trainingSet.input.shape[1]) * 0.6 - 0.3
                    # np.ones(self.trainingSet.input.shape[1])

        self.activation = Activation.getActivation(activation)
        self.activationPrime = Activation.getDerivative(activation)
        self.activationString = activation[0].upper() + activation[1:]

        self.erString = error

        if error == 'absolute':
            self.erf = erf.AbsoluteError()
        elif error == 'different':
            self.erf = erf.DifferentError()
        elif error == 'mse':
            self.erf = erf.MeanSquaredError()
        elif error == 'sse':
            self.erf = erf.SumSquaredError()
        elif error == 'bce':
            self.erf = erf.BinaryCrossEntropyError()
        elif error == 'crossentropy':
            self.erf = erf.CrossEntropyError()
        else:
            raise ValueError('Cannot instantiate the requested '
                             'error function: ' + error + 'not available')

    def train(self, verbose=False):
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

        # no bias
        input = normalized_input
        # with bias:
        # self.weight = np.append(self.weight, np.random.rand(1)) * 0.001
        # input = map(lambda x: np.append(x, [1]), normalized_input)

        # compute error gradient, plot error and update weights;
        epoch = 0

        self.initialize_plot()
        while epoch < self.epochs:
            d = np.ones(len(input))    # desired output
            o = map(self.fire, input)  # actual output

            epoch += 1
            err = self.erf.calculateError(d, np.array(o))
            dE_dy = self.erf.calculateErrorPrime(d, o)  # could be done in one go in the above method...

            self.update_plot(epoch, err)

            if err > 0.001:     # how to settle on a convergence threshhold?
                for i in range(0, len(input)):
                    dE_dx = dE_dy[i] * self.activationPrime(o[i])
                    self.updateWeights(input[i], dE_dx)

            # start cross validation
            if verbose:
                self.print_cross_validation_nfo(epoch)

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
        # if there is bias
        # return self.fire(np.append(testInstance, [1])) > 0.5
        # without bias
        return self.fire(testInstance) > 0.5

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

    def updateWeights(self, input, error_gradient):
        dE_dw = error_gradient * input
        self.weight = np.add(self.learningRate * dE_dw,
                             self.weight)

    def fire(self, input):
        """Fire the output of the perceptron corresponding to the input """
        return self.activation(np.dot(np.array(input), self.weight))

    def initialize_plot(self):
        plt.ion()  # for realtime (non blocking) plot updates
        plt.xlabel('epoch')
        plt.ylabel(self.erString + ' error')
        plt.title('Error Progression for Neuron \nwith ' +
                  self.activationString + ' Activation Function' +
                  ' on a Learning Rate of ' + str(self.learningRate))

    def update_plot(self, x, y, pause=0.05):
        plt.plot(x, y, 'b*-')   # add err value to runtime plot
        plt.draw()
        plt.pause(pause)  # display changes

    def print_cross_validation_nfo(self, epoch):
        validation_results = self.evaluate(self.validationSet)
        n_misses = np.sum(validation_results)

        accuracy_string = 'accuracy %(acc)02.02f' \
                          % {'acc': 100 * (1.0 - float(n_misses) / len(input))}
        logging.debug("(epoch %(epoch)02d) %(acc)s%%"
                      % {'epoch': epoch, 'acc': accuracy_string})
