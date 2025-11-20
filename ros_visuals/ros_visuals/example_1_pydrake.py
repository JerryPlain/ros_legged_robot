from pydrake.all import MathematicalProgram, Solve # import necessary modules from pydrake

"""
Example 1

Pydrake is a framework that is specialized in solving and simulating
optimization problems we will use it to solve the trajectory planning problem from the lecture.

This is the most basic example of how to solve a MathematicalProblem
Many more examples can be found in the drake tutorials section:

https://github.com/RobotLocomotion/drake/blob/master/tutorials/mathematical_program.ipynb

In this file:

Problem: find the smallest number that is bigger than 3.
"""

# 1. Define an instance of MathematicalProgram 
prog = MathematicalProgram() 

# 2. Add decision variables
x = prog.NewContinuousVariables(1)  # define one decision variable, e.g. x[0]

# 3. Add Cost function 
prog.AddCost(x.dot(x))

# 4. Add Constraints
prog.AddConstraint(x[0] >= 3)

# 5. Solve the problem 
result = Solve(prog)  # Call the solver (internally uses QP or NLP solvers)

# 6. Get the solution
if (result.is_success):
  print("Solution: " + str(result.GetSolution()))
    