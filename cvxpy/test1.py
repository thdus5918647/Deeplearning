import cvxpy as cp

# Create two scalar optimization variables.
x = cp.Variable()
y = cp.Variable()

print(x)
print(y)

# Create two constraints.
constraints = [x + y == 1,
               x - y >= 1]

# Form objective.
obj = cp.Minimize((x - y)**2)

# Form and solve problem.
prob = cp.Problem(obj, constraints)
prob.solve()  # Returns the optimal value.
print("status:", prob.status)
print("optimal value", prob.value)
print("optimal var", x.value, y.value)

# Replace the objective.
prob2 = cp.Problem(cp.Maximize(x+y), prob.constraints)
print("optimal value", prob2.solve())

# Replace the constraint (x+y==1).
constraints = [x+y<=3] + prob2.constraints[1:]
prob3 = cp.Problem(prob2.objective, constraints)
print("optimal valus", prob3.solve())

