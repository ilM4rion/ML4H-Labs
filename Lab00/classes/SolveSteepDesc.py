import numpy as np
import matplotlib.pyplot as plt
from .SolveMinProbl import SolveMinProbl

class SolveSteepDesc(SolveMinProbl):

    def run(self, Nit):
        self.Nit = Nit
        self.err = np.empty((0,2), dtype=float)

        w = np.random.rand(self.Nf,1)
        X = self.matr
        y = self.y
        hess = 2*X.T @ X # Hessian matrix does not depend on w, so can be computed before

        for i in range(Nit):
            grad = 2*X.T @ (X@w - y) # (-2*X.T*y + 2*X.T*w)
            den = (grad.T @ hess @ grad).item()
            if np.abs(den) < 1e-12:
                print(f"Too small {i}: {den}")
                break 
            alpha = (np.linalg.norm(grad)**2) / den

            w = w - alpha*grad
            sqerr = np.linalg.norm(X@w - y)**2
            self.err = np.append(self.err, np.array([[i,sqerr]]), axis=0)
            self.w_hat = w
            self.min = sqerr

    def plot_err(self, title, logy, logx):

        err = self.err
        plt.figure()

        if(logy==0 and logx==0):
            plt.plot(err[:,0], err[:,1])
        elif(logy==1 and logx==0):
            plt.semilogy(err[:,0], err[:,1])
        elif(logy==0 and logx==1):
            plt.semilogx(err[:,0], err[:,1])
        elif(logy==1 and logx==1):
            plt.loglog(err[:,0], err[:,1])

        plt.xlabel("n")
        plt.ylabel("e(n)")
        plt.title(title)
        plt.margins(0.01, 0.1)
        plt.grid()
        plt.show()
        return
        
