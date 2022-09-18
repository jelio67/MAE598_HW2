import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def f_x(x):
    x2 = x[0]
    x3 = x[1]
    f = 5 * x2 ** 2 + 12 * x2 * x3 - 8 * x2 + 10 * x3 ** 2 - 14 * x3 - 5
    return f


def g_x(x):
    x2 = x[0]
    x3 = x[1]
    g1 = 10 * x2 + 12 * x3 - 8
    g2 = 12 * x2 + 20 * x3 - 14
    g = np.array([g1, g2])
    return g


def phi_alpha(alpha, t, f, g):
    phi = f - t*g.transpose() @ g*alpha # can use @ operator instead of np.matmul
    return phi


def Inexact_Line_Search(XN, t, alpha, max_iter):
    iter = 0
    g = g_x(XN)
    f = f_x(XN)  # compute function value at current point
    g = g_x(XN)  # compute gradient at current point
    f_x_ag = f_x(XN - alpha * g)  # actual function value at step size alpha
    phi = phi_alpha(alpha, t, f, g)  # 1st order approx. of function value at step size alpha

    while f_x_ag > phi and iter < max_iter:
        f = f_x(XN)  # compute function value at current point
        g = g_x(XN)  # compute gradient at current point
        f_x_ag = f_x(XN - alpha * g)  # actual function value at step size alpha
        phi = phi_alpha(alpha, t, f, g)  # 1st order approx. of function value at step size alpha

        alpha = 0.5 * alpha

        iter += 1

    return alpha


def Gradient_Descent(X0, t, alpha0, eps, max_iter):
    results = []
    XN = X0
    f = f_x(XN)  # compute function value at initial point
    g = g_x(XN)  # compute gradient at initial point
    f_x_ag = f_x(XN - alpha0 * g)  # actual function value at step size alpha
    phi = phi_alpha(alpha0, t, f, g)  # 1st order approx. of function value at step size alpha
    iter = 0

    results.append([iter, f, g[0], g[1], f_x_ag, phi, alpha0, np.linalg.norm(g)])

    while np.linalg.norm(g) > eps and iter < max_iter:

        alpha = Inexact_Line_Search(XN, t, alpha0, max_iter)

        XN = XN - alpha*g

        f = f_x(XN)  # compute function value at next point
        g = g_x(XN)  # compute gradient at next point
        f_x_ag = f_x(XN - alpha * g)  # actual function value at step size alpha
        phi = phi_alpha(alpha, t, f, g)  # 1st order approx. of function value at step size alpha

        iter += 1

        results.append([iter, f, g[0], g[1], f_x_ag, phi, alpha0, np.linalg.norm(g)])

    return XN, results

def Convergence(f_list, f_star, X0):
    X2_0 = int(X0[0])
    X3_0 = int(X0[1])
    error = np.zeros(len(f_list))
    for i in range(0, len(f_list)):
        error[i] = abs(f_list[i] - f_star)

    plt.figure()
    plt.plot(error)
    plt.ylabel(r'|$f_k$-$f^{*}$|')
    plt.yscale('log')
    plt.ylim([1e-13, 10])
    plt.xlabel('Iteration #, k')
    plt.xlim([0, 90])
    plt.savefig('.\\P2b_GD_Results\Convergence_X2_0='+str(X2_0)+'_X3_0='+str(X3_0)+'.jpg', bbox_inches='tight', dpi=300)





# set params
t = 0.5
alpha = 1
eps = 1e-6
max_iter = 1000

# initial guess
x2_0 = 100
x3_0 = -100
X0 = np.array([[x2_0], [x3_0]])

# Calculate
XN, results = Gradient_Descent(X0, t, alpha, eps, max_iter)
Results = pd.DataFrame(results, columns=['iter', 'f', 'g[0]', 'g[1]', 'f_x_ag', 'phi', 'alpha', 'Norm(g)'])
Results.to_csv('.\\P2b_GD_Results\Debug_Results_X2_0='+str(x2_0)+'_X3_0='+str(x3_0)+'.csv', index=False)

# plot convergence
f_list = np.array(Results['f'])
x_star = np.array([[-1/7], [11/14]])
f_star = f_x(x_star)
Convergence(f_list, f_star, X0)

# print results
Total_Iterations = len(f_list)+1
print(XN)
print('\n'+str(Total_Iterations))
