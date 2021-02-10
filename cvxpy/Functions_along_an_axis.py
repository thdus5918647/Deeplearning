import cvxpy
import numpy

X = cvxpy.Variable((5,4))
col_sums1 = cvxpy.sum(X, axis=0, keepdims=True)
col_sums2 = cvxpy.sum(X, axis=0)
row_sums = cvxpy.sum(X, axis=1)

print(col_sums1.size)
print(col_sums2.size)
print(row_sums.size)
