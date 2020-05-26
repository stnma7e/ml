
import numpy as np
from sklearn.base import RegressorMixin
from sklearn.preprocessing import PolynomialFeatures

import numpy.linalg as la
import pandas as pd

from sklearn.base import RegressorMixin
from sklearn.preprocessing import PolynomialFeatures

class LinReg(RegressorMixin):
    def __init__(self, N: int):
        self.N = N
        self.poly_features = PolynomialFeatures(degree=self.N, include_bias=True)

    def transform(self, X: np.array, y: np.array):
        return X

    def fit_transform(self, X: np.array, y: np.array):
        self.fit(X, y)
        return X

    def predict(self, X: np.array):
        X_b = self.poly_features.fit_transform(X)
        return X_b.dot(self.theta)

class LeastSq(LinReg):
    def __init__(self, N:int):
        super(LeastSq, self).__init__(N)

    def fit(self, X: np.array, y: np.array):
        X_b = self.poly_features.fit_transform(X)
        self.theta = la.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        try:
            pass
            # self.theta = la.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        except np.linalg.LinAlgError:
            # probably a singular matrix error
            self.theta = np.randn(self.N+1,1)

class GD(LinReg):
    def __init__(self, N:int, eta:float=0.01):
        super(GD, self).__init__(N)
        self.eta = eta

    def fit(self, X: np.array, y: np.array):
        M = len(X)
        X_b = self.poly_features.fit_transform(X)

        self.theta = np.random.randn(self.N+1, 1)
        for i in range(1000):
            dMSE = 2/M * X_b.T.dot(X_b.dot(self.theta) - y)
            self.theta -= self.eta * dMSE

class SGD(LinReg):
    def __init__(self, N:int, batch_size:int=1, n_epochs:int=10, t0:float=1.0, t1:float=50):
        super(SGD, self).__init__(N)
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.t0 = t0
        self.t1 = t1

    def learning_schedule(self, t):
        return self.t0 / (t + self.t1)

    def fit(self, X: np.array, y: np.array):
        M = len(X)
        X_b = self.poly_features.fit_transform(X)
        self.theta = np.random.randn(self.N+1, 1)

        for epoch in range(self.n_epochs):
            for i in range(M // self.batch_size):
                j = np.random.randint(M)
                xj = X_b[j:min(j+self.batch_size, M - 1)]
                yj = y[j:min(j+self.batch_size, M - 1)]
                dMSE = 2 * xj.T.dot(xj.dot(self.theta) - yj)
                eta = self.learning_schedule(epoch * M + i)
                self.theta -= eta * dMSE
