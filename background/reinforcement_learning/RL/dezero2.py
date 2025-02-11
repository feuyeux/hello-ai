import numpy as np
from dezero import Variable

# 罗森布罗克函数的每个等高线大致呈抛物线形，其全局最小值也位在抛物线形的山谷中（香蕉型山谷）。
# 很容易找到这个山谷，但由于山谷内的值变化不大，要找到全局的最小值相当困难


def rosenbrock(x0, x1):
    y = 100 * (x1 - x0 ** 2) ** 2 + (x0 - 1) ** 2
    return y


x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))
# learning rate
lr = 0.001
iters = 10000

for i in range(iters):
    y = rosenbrock(x0, x1)

    x0.cleargrad()
    x1.cleargrad()
    y.backward()

    x0.data -= lr * x0.grad.data
    x1.data -= lr * x1.grad.data

print(x0, x1)
