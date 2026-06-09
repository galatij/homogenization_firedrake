from firedrake import *
from cell_problem import CellProblem
from homogenization_problem import HomogenizationProblem

# @TODO:
# 1. implement CellProblem._build_solver() @DONE (check)
# 2. impose zero mean correctors and identity for the first N_uu @ DONE (check)
# 3. solve for a = 1,2,3 @DONE
# 4. choose g appropriately (analysis)
# 5. output and debug
# 6. comparison with standard elasticity


def main():
    n = 15
    mu_fun = lambda x,y,z: conditional(z > 0.5, Constant(0.2), Constant(0.8))
    f_fun = lambda x,y,z: as_vector((0., 0., sin(2*pi*z)))

    cell_pb = CellProblem(n, mu_fun, f_fun)
    homogenization_pb = HomogenizationProblem(cell_pb, 2)
    homogenization_pb.run()

    return 0

if __name__ == "__main__":
    main()