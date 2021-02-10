import cvxpy as cp

x = cp.Variable(pos=True)
y = cp.Variable(pos=True)

constant = cp.Constant(2.0)
monomial = constant * x * y
posynomial = monomial + (x ** 1.5) * (y ** -1)

print(monomial.is_dgp())
assert posynomial.is_dgp()
