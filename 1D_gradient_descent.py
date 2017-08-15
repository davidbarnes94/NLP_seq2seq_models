import numpy as np
import matplotlib.pyplot as plt

x_old = 0
x_new = 6
eps = 0.001
precision = 0.00001
x = np.linspace(3.5, 4.5, 1000)


def f_derivative(x):
    return 4*(x**3) - 9*(x**2)

def f(x):
    return x**4 - 3 * x**3 + 2

def build_tangent_line(m, x, x_0 = 4, y_0 = f(4)):
    return m*x + (y_0 - m*x_0)


while abs(x_new - x_old) > precision:
    x_old = x_new
    x_new = x_old - eps*f_derivative(x_old)
    #print x_new

y = f(x)
#build_tangent_line(208, x, 3, 4)
plt.plot(x, y)
plt.plot(x, build_tangent_line(f_derivative(4), x), label = '$m$')
plt.plot(x, build_tangent_line(f_derivative(4) + 10, x), label = '$m+10$')
plt.plot(x, build_tangent_line(f_derivative(4) - 10, x), label = '$m-10$')
plt.plot(x, build_tangent_line(f_derivative(4) + 100, x), label = '$m+100$')
plt.legend()
plt.show()