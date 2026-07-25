from firedrake import *
from cell_problem import CellProblem
from homogenization_problem import HomogenizationProblem
from geometry import *
from effective_problem import *
from check_effective import *

# @TODO:
# 1. implement CellProblem._build_solver() @DONE (check)
# 2. impose zero mean correctors and identity for the first N_uu @ DONE (check)
# 3. solve for a = 1,2,3 @DONE
# 4. choose g appropriately (analysis)
# 5. output and debug
# 6. comparison with standard elasticity


def main():
    # DATA FOR THE LOCAL PROBLEM
    n = 20
    data = {
        "R": 5,
        "t": 2.5,
        "l": 25
    }
    order = 2

    mu_fun = lambda x,y,z: conditional(inside_dark(x,y,z, data), Constant(1e8), Constant(1e10))
    lmbda_fun = lambda x,y,z: conditional(inside_dark(x,y,z, data), Constant(0.0), Constant(0.0))
    # mu_fun = lambda x,y,z: conditional(y >= 0.5, Constant(0.001), Constant(0.9))
    # lmbda_fun = lambda x,y,z: conditional(y >= 0.5, Constant(0.0), Constant(0.0))

    # SOLVE THE LOCAL PERIODIC PROBLEM AND COMPUTE EFFECTIVE COEFFICIENTS
    cell_pb = CellProblem(n, mu_fun, lmbda_fun)
    homogenization_pb = HomogenizationProblem(cell_pb, order)

    Ceff, tCeff = homogenization_pb.run()

    # CHECK EFFECTIVE COEFFICIENTS
    results = check_effective_tensors(
        Ceff[2],
        tCeff[2],
        CST=cell_pb.CST_np,
        verbose=True,
        plot=True
    )

    # SOLVE GLOBAL PROBLEM WITH EFFECTIVE COEFFICIENTS
    f_fun = lambda x,y,z: as_vector((0., -5., 0.))
    coarse_mesh = BoxMesh(40,20,1,40,10,1)
    eff_pb = EffectiveElasticityProblem(coarse_mesh, Ceff, tCeff, f_fun)
    eff_pb.solve()


    return 0

if __name__ == "__main__":
    main()



# n = 120
# R = 5
# t = 0.5
# l = 25
# a = 2*np.sqrt(l*l/4 + R*R)
# xc = a/2
# yc = a/2
# theta = np.arctan(l/(2*R)) + pi/4
