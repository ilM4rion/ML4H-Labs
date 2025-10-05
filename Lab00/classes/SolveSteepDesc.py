import numpy as np
import matplotlib.pyplot as plt
from .SolveMinProbl import SolveMinProbl

class SolveSteepDesc(SolveMinProbl):

    def run(self, Nit):
        self.Nit = Nit
        self.err = np.empty((0,2), dtype=float)

        w = np.random.rand(self.Nf,1)
        X = self.matrix
        y = self.y
        hess = 2*X.T @ X # Hessian matrix does not depend on w, so can be computed before

        for i in range(Nit):
            grad = 2*X.T @ (X@w - y) # (-2*X.T*y + 2*X.T*w)
            alpha = float((np.linalg.norm(grad)**2)/(grad.T @ hess @ grad))
            w = w - alpha*grad
            sqerr = np.linalg.norm(X@w - y)**2
            self.err = np.append(self.err, np.array([[i,sqerr]]))
            self.w_hat = w
            self.min = sqerr