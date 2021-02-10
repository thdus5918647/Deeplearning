import cvxpy as cp

# DGP requires Variables to be declared positive via `pos=True`.
x = cp.Variable(pos=True)
y = cp.Variable(pos=True)
z = cp.Variable(pos=True)

objective_fn = x * y * z
constraints = [
          4 * x * y * z + 2 * x * z <= 10, x <= 2*y, y <= 2*x, z >= 1]
assert objective_fn.is_log_log_concave()
assert all(constraint.is_dgp() for constraint in constraints)
problem = cp.Problem(cp.Maximize(objective_fn), constraints)
assert problem.is_dgp()

# All Variables must be declared as positive for an Expression to be DGP.
w = cp.Variable()
objective_fn = w * x * y 
assert not objective_fn.is_dgp()
problem = cp.Problem(cp.Maximize(objective_fn), constraints)
assert not problem.is_dgp()
