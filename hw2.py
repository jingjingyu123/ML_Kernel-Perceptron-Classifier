#!/usr/bin/env python3

import math
import numpy as np
from time import time
from collections import Counter

class Data:
    def __init__(self):
        self.features = []	# list of lists (size: number_of_examples x number_of_features)
        self.labels = []	# list of strings (lenght: number_of_examples)

def read_data(path):
    data = Data()
    # TODO: function that will read the input file and store it in the data structure
    # use the Data class defined above to store the information
    data = Data()
    f = open(path,'r')
    out = f.readlines()
    for line in out:
            x = line.split(",")
            # data.features.append(x[0:4])
            temp_feature_arr = []
            for i in range(4):
                temp_feature_arr.append(float(x[i]))
            data.features.append(temp_feature_arr)
            if x[4] == "Iris-setosa\n":
                data.labels.append(1.0)
            elif x[4] == "Iris-setosa":
                data.labels.append(1.0)
            else:
                data.labels.append(-1.0)
    f.close()
    return data

def dot_kf(u, v):
    """
    The basic dot product kernel returns u*v.

    Args:
        u: list of numbers
        v: list of numbers

    Returns:
        u*v
    """
    # TODO: implement the kernel function

    counter = 0
    if len(u)==len(v):
        for i in range(len(u)):
            counter = counter + (u[i]*v[i])
    return counter

def poly_kernel(d):
    """
    The polynomial kernel.

    Args:
        d: a number

    Returns:
        A function that takes two vectors u and v,
        and returns (u*v+1)^d.
    """
    def kf(u, v):
        # TODO: implement the kernel function
        dp = dot_kf(u, v)
        return (dp+1)**d
    return kf

def exp_kernel(s):
    """
    The exponential kernel.

    Args:
        s: a number

    Returns:
        A function that takes two vectors u and v,
        and returns exp(-||u-v||/(2*s^2))
    """
    def kf(u, v):
        # TODO: implement the kernel function
        # diff = []
        norm_sqr = 0
        for i in range(len(u)):
            # diff.append(u[i]-v[i])
            norm_sqr += (u[i]-v[i])**2
        norm = math.sqrt(norm_sqr)
        return math.exp(-norm/(2*(s**2)))
    return kf

class Perceptron:
    def __init__(self, kf, lr):
        """
        Args:
            kf - a kernel function that takes in two vectors and returns
            a single number.
        """
        self.MissedPoints = []
        self.MissedLabels = []
        self.kf = kf
        self.lr = lr
        # self.alpha = []
        # self.data = Data()

    def train(self, data):
        # TODO: Main function - train the perceptron with data
        self.data = data
        self.alpha = np.zeros(len(self.data.labels))
        n_samples = len(self.data.features)
        n_features = len(self.data.features[0])


        # Gram matrix
        # K = np.zeros((n_samples, n_samples))
        # for i in range(n_samples):
        #     for j in range(n_samples):
        #         K[i,j] = self.kf(self.data.features[i], self.data.features[j])
        

        converged = False
        while converged == False:
            converged = True
            for i in range(n_samples):
                if self.update(self.data.features[i], self.data.labels[i]):
                    self.alpha[i] += 1.0
                    print("add 1 to alpha "+str(i)+" now alpha = "+str(self.alpha[i]))
                    converged = False

        return

    def update(self, point, label):
        """
        Updates the parameters of the perceptron, given a point and a label.

        Args:
            point: a list of numbers
            label: either 1 or -1

        Returns:
            True if there is an update (prediction is wrong),
            False otherwise (prediction is accurate).
        """
        # TODO
        temp = 0
        for j in range(len(self.data.labels)):
            temp = temp + self.alpha[j]*self.data.labels[j]*self.kf(point, self.data.features[j])
        pred = label * temp
        is_mistake = (pred <= 0)
        return is_mistake

    def predict(self, point):
        """
        Given a point, predicts the label of that point (1 or -1).
        """
        # TODO
        temp = 0
        for j in range(len(self.data.labels)):
            temp = temp + self.alpha[j]*self.data.labels[j]*self.kf(point, self.data.features[j])
        pred_temp = temp
        if pred_temp <= 0:
            pred = -1
        else:
            pred = 1
        # print(pred_temp)
        return pred

    def test(self, data):
        predictions = []
        # TODO: given data and a perceptron - return a list of integers (+1 or -1).
        # +1 means it is Iris Setosa
        # -1 means it is not Iris Setosa
        for i in range(len(data.labels)):
            predictions.append(self.predict(data.features[i]))
        return predictions


# Feel free to add any helper functions as needed.
if __name__ == '__main__':
    perceptron = Perceptron(exp_kernel(20), 0)
    data = read_data("hw2_train.txt")
    perceptron.train(data)
    test_data = read_data("hw2_test.txt")
    pred = perceptron.test(test_data)


    corr_counter = 0
    for i in range(len(test_data.labels)):
        if test_data.labels[i] == pred[i]:
            corr_counter = corr_counter + 1
    
    print("accuracy: "+ str(corr_counter)+ " out of "+str(len(test_data.labels)))
    print(pred)
    print(test_data.labels)

