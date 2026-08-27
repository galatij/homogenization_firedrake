from firedrake import *
from cell_problem import CellProblem
from homogenization_problem import HomogenizationProblem
from geometry import *
from effective_problem import *
from ho_effective_problem import *
from check_effective import *
from fine_scale import *

# @TODO:
# 1. understand the symmetry of the effective tensor
# 2. implement h.o. stresses @DOING...
# 3. improve code
# 4. improve geometry
# 5. run simulations and plots


def main():
    # DATA FOR THE LOCAL PROBLEM
    n = 50
    data = {
        "R": 5,
        "t": 2.5,
        "l": 25
    }
    dim = 2
    order = 2
    loadx = -5
    loady = 0
    mu_dark = 1e10
    mu_light = 1e7
    lmbda_dark = 0
    lmbda_light = 0

    mu_fun = lambda x,y,z: conditional(inside_dark(x,y,z, data), Constant(mu_dark), Constant(mu_light))
    lmbda_fun = lambda x,y,z: conditional(inside_dark(x,y,z, data), Constant(lmbda_dark), Constant(lmbda_light))
    # mu_fun = lambda x,y,z: conditional(y >= 0.5, Constant(0.001), Constant(0.9))
    # lmbda_fun = lambda x,y,z: conditional(y >= 0.5, Constant(0.0), Constant(0.0))

    # SOLVE THE LOCAL PERIODIC PROBLEM AND COMPUTE EFFECTIVE COEFFICIENTS
    cell_pb = CellProblem(n, mu_fun, lmbda_fun, dim)
    homogenization_pb = HomogenizationProblem(cell_pb, order)

    print("\nComputing effective coefficients...")
    Ceff, tCeff = homogenization_pb.run()
    print(" done.")

    # # CHECK EFFECTIVE COEFFICIENTS
    # results = check_effective_tensors(
    #     dim,
    #     Ceff[2],
    #     tCeff[2],
    #     CST=cell_pb.CST_np,
    #     verbose=True,
    #     plot=True
    # )

    # SOLVE GLOBAL PROBLEM WITH EFFECTIVE COEFFICIENTS
    f_vec = np.zeros(dim)
    f_vec[0] = loadx
    f_vec[1] = loady
    f_fun = lambda x,y,z: as_vector(f_vec)

    coarse_mesh = BoxMesh(80,40,1,20,10,1) if dim==3 else RectangleMesh(80,40,20,10)
    eff_pb = EffectiveElasticityProblem(coarse_mesh, Ceff, tCeff, f_fun, dim)
    eff_pb.solve()
    
    # # eff_pb = HoEffectiveElasticityProblem(coarse_mesh, Ceff, tCeff, f_fun, dim, order)
    # # eff_pb.solve()

    # FINE SCALE PROBLEM
    fs_geom = {
        "nx": 20,
        "ny": 10,
        "nz": 1,
        "ncell": 50,
        "Lcell": 1
    }

    mu_fun = lambda x, y, z: conditional(
            inside_dark(
                x,y,z,
                data,
                fine_scale=True,
                ncell=(fs_geom["nx"],fs_geom["ny"],fs_geom["nz"])
            ),
            Constant(mu_dark),
            Constant(mu_light)
        )

    fs_pb = FineScaleProblem(mu_fun, lmbda_fun, f_fun, dim, fs_geom)
    fs_pb.solve()


    return 0

if __name__ == "__main__":
    main()

