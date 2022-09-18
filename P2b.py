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


def Gradient_Descent(X0, t, alpha, eps, max_iter):
    results = []
    XN = X0
    f = f_x(XN)  # compute function value at initial point
    g = g_x(XN)  # compute gradient at initial point
    iter = 0

    results.append([iter, f, g, alpha])

    while np.linalg.norm(g) > eps and iter < max_iter:

        alpha = Inexact_Line_Search(XN, t, alpha, max_iter)

        XN = XN - alpha*g

        f = f_x(XN)  # compute function value at current point
        g = g_x(XN)  # compute gradient at current point

        iter += 1

        results.append([iter, f, g, alpha])

    return XN, results










# set params
t = 0.5
alpha = 1
eps = 1e-6
max_iter = 1000

# initial guess
x2_0 = 0
x3_0 = 0
X0 = np.array([[x2_0], [x3_0]])

# Calculate
XN, results = Gradient_Descent(X0, t, alpha, eps, max_iter)
Results = pd.DataFrame(results, columns=['iter', 'f', 'g', 'alpha'])
Results.to_csv('Results.csv', index=False)
print(XN)

# x_star = np.array([[-1/7], [11/14]])
# f_star = f_x(x_star)
# c = np.zeros(len(results))
# for i in range(0, len(c)-1):
#     c[i] = abs(f_list[i+1]-f_star)/abs(f_list[i]-f_star)

# plt.figure()
# plt.plot(c)
# plt.yscale('log')
# plt.ylim([1e-6, 1e2])
# plt.savefig('test.png', bbox_inches='tight', dpi=300)




