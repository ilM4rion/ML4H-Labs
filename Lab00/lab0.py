import numpy as np
from classes.SolveLLS import SolveLLS
from classes.SolveGrad import SolveGrad
from classes.SolveSteepDesc import SolveSteepDesc

np.random.seed(104)

Np = 100 # number of rows
Nf = 4 # number of columns
X = np.random.randn(Np,Nf) # matrix/Ndarray X
w = np.random.randn(Nf, 1) # true vector w
y = X@w # column vector y
m = SolveLLS(y,X)
m.run()
m.print_result('LLS')
m.plot_what('LLS')

# hyperparameter
Nit = 1000  # number of iterations
gamma = 1e-5 # learning rate
epsilon = 1e-6 # stopping condition
max_iter=100000 # safe stopping condition

# Gradient Algorithm using iteration as stopping condition
g = SolveGrad(y,X)
g.run(gamma, Nit)
logx = 0
logy = 0
g.plot_what("Gradient Algorithm")
g.plot_err("Gradient Algorithm: square error", logy, logx)


# Gradient Algorithm using epsilon as stopping condition
g_2 = SolveGrad(y,X)
g_2.run2(gamma, epsilon, max_iter)
logx = 0
logy = 0
g_2.plot_what("Gradient Algorithm with epsilon")
g_2.plot_err("Gradient Algorithm with epsilon: square error", logy, logx)


steep = SolveSteepDesc(y,X)
steep.run(Nit)
logx = 0
logy = 0
steep.plot_what("Steepest Descent")
steep.plot_err("Steepest Descent: square error", logy, logx)
