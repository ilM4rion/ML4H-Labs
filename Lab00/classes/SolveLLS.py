import numpy as np
from .SolveMinProbl import SolveMinProbl

class SolveLLS(SolveMinProbl):

    def run(self):
        X = self.matr  # retrieve the know matrix X
        y = self.y  # retrieve the know vector y
        w_hat = np.linalg.inv(X.T@X)@(X.T@y) # w_hat = (X^T*X)^(-1)*(X^T*y)
        self.w_hat = w_hat
        self.min = np.linalg.norm(X@w_hat-y)**2 # square error norm
        return
