import numpy as np
import matplotlib.pyplot as plt
from .SolveMinProbl import SolveMinProbl

class SolveGrad(SolveMinProbl):

    def run(self, gamma, Nit): # hyperparamenters gamma and number of iterations (stopping condition)
        self.err = np.empty((0,2), dtype=float) # empty array with two columns
        self.gamma = gamma
        self.Nit = Nit
        X = self.matr #retrives X from upper class
        y = self.y
        w = np.random.rand(self.Nf, 1) # random initialization of the weight vector w

        for i in range(Nit):
            grad = 2*X.T@(X@w-y) # gradient of the current value of w  the formula is -2X.T*y+2X.T*X*w
            w = w-gamma*grad # update the value of w
            sqerr = np.linalg.norm(X@w-y)**2
            self.err = np.append(self.err, np.array([[i,sqerr]]), axis=0)
            self.w_hat = w # store w in w_hat
            self.min = sqerr

    # hyperparamemeters gamma and epsilon, max_iter is used as safe stopping condition in case the main condition is not verified
    def run2(self, gamma, epsilon, max_iter): 
        self.err = np.empty((0,2), dtype=float)
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_iter = max_iter

        X = self.matr
        y= self.y
        w = np.random.rand(self.Nf, 1) # Nf number of features --> rows of w
        self.w_hat = w
        self.min = np.linalg.norm(X@w-y)**2

        for i in range(max_iter):
            grad = 2*X.T@(X@w-y)
            w_2 = w-gamma*grad # w_2 is w(i+1) while w is w(i)

            # calculate the norm of the difference between w_2 and w
            norm_diff = np.linalg.norm(w_2-w)

            # calculate the square error
            sqerr = np.linalg.norm(X@w_2-y)**2
            self.err = np.append(self.err, np.array([[i,sqerr]]), axis=0)
            w = w_2
            self.w_hat = w
            self.min = sqerr

            # Check the stopping condition
            if norm_diff < epsilon:
                print("Converged after ",i, " iterations.")
                break

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
            


