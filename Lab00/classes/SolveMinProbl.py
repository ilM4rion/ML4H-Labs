import numpy as np
import matplotlib.pyplot as plt

class SolveMinProbl:
    def __init__(self, y=np.ones((3,)), X=np.eye(3)): # X matrice identità 3x3
        self.matr = X #matric X (known)
        self.y = y # column vector y (know)
        self.Np = X.shape[0] # number of rows
        self.Nf = X.shape[1] # number of columns
        self.w_hat = np.zeros((self.Nf, 1), dtype=float) # column vector w_hat to be found
        self.min = 0 # square norm of the error
        return
    
# methods
    def plot_what(self, title):
        w_hat = self.w_hat
        n = np.arange(self.Nf)
        plt.figure() 
        plt.plot(n,w_hat)
        plt.xlabel("n")
        plt.ylabel("w_hat(n)")
        plt.title(title)
        plt.grid()
        plt.show()
        return
    
    def print_result(self, title):
        print(title, " :")
        print("the optimum wight vector is: ")
        print(self.w_hat.T)
        return