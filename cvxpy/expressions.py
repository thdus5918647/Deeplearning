import numpy
import cvxpy as cp 

X = cp.Variable((5,4))
A = numpy.ones((3,5))

# Use expr.shape to get the dimensions.
print("dimensions of X:" , X.shape)
print("size of X:", X.size)
print("number of dimensions:", X.ndim)
print("dimensions of sum(X):", cp.sum(X).shape)
print("dimensions of A @ X:", (A @ X).shape)

# ValueError raised for invalid dimensions.
try:
    A+X
except ValueError as e:
    print(e)

