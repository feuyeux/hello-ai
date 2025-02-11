import numpy as np
# 修改  dezero/transforms.py line 154
# def __init__(self, dtype=np.int)
# => def __init__(self, dtype=np.int32)
from dezero import Variable
import dezero.functions as F

# Inner products
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
a, b = Variable(a), Variable(b)  # Optional
c = F.matmul(a, b)
print(1*4+2*5+3*6)
print(c)
print("====")
# Matrix product
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
print(1*5+2*7, 1*6+2*8)
print(3*5+4*7, 3*6+4*8)
c = F.matmul(a, b)
print(c)
