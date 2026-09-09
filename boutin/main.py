from firedrake import *
from geometry import *
from effective_problem import EffectiveProblem
from effective_problem_linear import EffectiveProblemLinear
from check_effective import *
from fine_scale import *
from cell_problem import CellProblem

def main():
    solve_hom = True
    solve_eff = True
    solve_fs = False
    check_coeff = False
    is_per = True

    mirror_structure = True

    # DATA FOR THE LOCAL PROBLEM
    n = 50
    data = {
        "R": 5,
        "t": 2.5,
        "l": 25,
        "mirror": mirror_structure
    }

    dim = 2
    order = 4
    loadx = 0
    loady = -5
    mu_dark = 1e9
    mu_light = 1e7
    lmbda_dark = 0
    lmbda_light = 0

    mu_fun = lambda x,y,z: conditional(inside_dark(x,y,z, data), Constant(mu_dark), Constant(mu_light))
    lmbda_fun = lambda x,y,z: conditional(inside_dark(x,y,z, data), Constant(lmbda_dark), Constant(lmbda_light))
    # mu_fun = lambda x,y,z: conditional(y >= 0.5, Constant(0.001), Constant(0.9))
    # lmbda_fun = lambda x,y,z: conditional(y >= 0.5, Constant(0.0), Constant(0.0))

    if solve_hom:
        # SOLVE THE LOCAL PERIODIC PROBLEM AND COMPUTE EFFECTIVE COEFFICIENTS
        print("\nComputing effective coefficients...")
        cell_pb = CellProblem(n, mu_fun, lmbda_fun, dim, order, output_dir="output/primal")
        Ceff = cell_pb.solve_all()        
        print(" done.")

    # SOLVE GLOBAL PROBLEM WITH EFFECTIVE COEFFICIENTS
    f_vec = np.zeros(dim)
    f_vec[0] = loadx
    f_vec[1] = loady
    f_fun = lambda x,y,z: as_vector(f_vec)

    if solve_eff:
        coarse_mesh = BoxMesh(64,64,1,16,16,1) if dim==3 else RectangleMesh(64,64,16,16)
        periodic_coarse_mesh = PeriodicBoxMesh(64,64,1,16,16,1) if dim==3 else PeriodicRectangleMesh(64,64,16,16)
        ell = 1
        eff_pb = EffectiveProblemLinear(periodic_coarse_mesh, ell, Ceff, f_fun, dim, order, is_per)
        eff_pb.solve()

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

    if solve_fs:
        fs_pb = FineScaleProblem(mu_fun, lmbda_fun, f_fun, dim, fs_geom)
        fs_pb.solve()


    return 0

if __name__ == "__main__":
    main()

