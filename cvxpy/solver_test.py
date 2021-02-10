import cvxpy as cp

# Solving a problem with different solvers.
x = cp.Variable(2)
obj = cp.Minimize(x[0] + cp.norm(x, 1))
constraints = [x >= 2]
prob = cp.Problem(obj, constraints)

# Solve with OSQP.
prob.solve(solver=cp.OSQP)
print("optimal value with OSQP:", prob.value)

# Solve with ECOS.
prob.solve(solver=cp.ECOS)
print("optimal value with ECOS:", prob.value)

# Solve with CVXOPT.
prob.solve(solver=cp.CVXOPT)
print("optimal value with CVXOPT:", prob.value)

# Solve with SCS.
prob.solve(solver=cp.SCS)
print("optimal value with SCS:", prob.value)

# Solve with GLPK.
prob.solve(solver=cp.GLPK)
print("optimal value with GLPK:", prob.value)

# Solve with GLPK_MI.
prob.solve(solver=cp.GLPK_MI)
print("optimal value with GLPK_MI:", prob.value)

# Solve with GUROBI.
prob.solve(solver=cp.GUROBI)
print("optimal value with GUROBI:", prob.value)

# Solve with MOSEK.
prob.solve(solver=cp.MOSEK)
print("optimal value with MOSEK:", prob.value)

# Solve with CBC.
prob.solve(solver=cp.CBC)
print("optimal value with CBC:", prob.value)

# Solve with CPLEX.
prob.solve(solver=cp.CPLEX)
print("optimal value with CPLEX:", prob.value)

# Solve with NAG.
prob.solve(solver=cp.NAG)
print("optimal value with NAG:", prob.value)

# Solve with SCIP.
prob.solve(solver=cp.SCIP)
print("optimal value with SCIP:", prob.value)

# Solve with XPRESS
prob.solve(solver=cp.XPRESS)
print("optimal value with XPRESS:", prob.value)
