import cvxpy as cp

x = cp.Variable(pos=True)
y = cp.Variable(pos=True)

constant = cp.Constant(2.0)
monomial = constant * x * y
posynomial = monomial + (x ** 1.5) * (y ** -1)
reciprocal = posynomial ** -1
unknown = reciprocal + posynomial

print(constant.log_log_curvature)
print(monomial.log_log_curvature)
print(posynomial.log_log_curvature)
print(reciprocal.log_log_curvature)
print(unknown.log_log_curvature)

print(constant.is_log_log_constant())
print(monomial.is_log_log_affine())
print(posynomial.is_log_log_convex())
print(reciprocal.is_log_log_concave())
