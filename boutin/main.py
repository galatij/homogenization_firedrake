from firedrake import *
from geometry import *
from cell_problem import CellProblem
from effective_problem_linear import EffectiveProblemLinear
from effective_problem_ho import EffectiveProblemHo
from fine_scale import *
from convergence import *
from numerical_convergence import *
from phunpeng import *
# from check_effective import *
import numpy as np


# TODO: 
#  - improve 66 mesh quality
# OK compute error w.r.t. most refined mesh
# OK compute twoscale error wrt finest microstructure (NOTE: nothing to do)
#  - try H1 error with U_RS instead of grad(U_eff)
#  - compute errors between finescale and effective (boutin) and effective (first order)
#  - understand how to properly compute L2, H1, average errors between solutions with different meshes

def main():

    data = {
        "dim": 2,                                       # dimension of the problem
        "order": 4,                                     # order of the effective equation
        "output_dir": "output/cellpb",
        "flags": {
            "solve_hom": False,
            "solve_eff": False,
            "solve_fs": True,
            "check_coeff": False,
            "convergence_test": False,
            "numerical_convergence_test": True,
            "is_per": True,
            "mirror_structure": False,
            "microstructure": "X",                       # "X" / "66" / "layered"
            "ho_type": "strain_gradient",                 # "strain_gradient" / "linear"
            "RSgrad_type": "I" ,
            "gmsh": True
        },
        "coefficients": {
            "loadx": -5,
            "loady": -5,
            "loadz": -5,
            "mu_dark": 1e10,
            "mu_light": 1e6,
            "lmbda_dark": 0,
            "lmbda_light": 0
        },
        "geometry": {
            "nref_cell": 16,                            # number of refinements
            "chiral": {
                "R": 7,   #5                              # radius
                "t": 2.5,   # 2.5                            # thikness of the ligaments
                "l": 25,                                # length of a ligament
                "mirror": False,                        # boolean
                "beta": np.pi/4                         # rotation angle of the ligaments
            },
            "crossed": {
                "t": 0.1,    # 0.15                          # thikness of the fibers
                "l_micro": 1                            # length of the domain
            },
            "fs_geom": {
                "nx_macro": 2,                         # number of microdomains (cell) along x
                "ny_macro": 2,                          # number of microdomains (cell) along y
                "nz_macro": 1,                          # number of microdomains (cell) along z
                "n_micro_fs": 1,                          # nref in a single microdomain (cell)
                "n_micro_eff": 1,
                "Lx_macro": 8,                         # length of macroscopic domain along x
                "Ly_macro": 8,                          # length of macroscopic domain along y
                "Lz_macro": 1,                          # length of macroscopic domain along z
            }
        },
        "convergence": {
            "nx_macro_values": [1, 2, 4, 8, 16],
            "ny_macro_values": [1, 2, 4, 8, 16],
            "reference": "fine_scale",
            "numerical_refinements": 5,
            "current_level": 0
        }
    }

    if data["flags"]["numerical_convergence_test"]:
        mu_dark_list = [1e7,1e8,1e9,1e10]
        for muuu in mu_dark_list:
            data["coefficients"]["mu_dark"] = muuu
            results = run_fem_convergence(
                data,
                solve_case
            )
    elif data["flags"]["convergence_test"]:
        mu_dark_list = [1e7,1e8,1e9,1e10]
        for muuu in mu_dark_list:
                
            data["coefficients"]["mu_dark"] = muuu

            nx_values = homogenization_sequence(2,16)

            conv = HomogenizationConvergenceTest(
                data,
                nx_macro_values=nx_values,
                ny_macro_values=nx_values,
                output_dir="output/convergence",
            )

            results = conv.run(solve_case)
            conv.print_table()
            conv.plot_convergence(save=True, show=True)
            conv.plot_relative_errors(save=True, show=True)
    else:
        U_fs, U_eff, U_RStrain, U_Lagrange = solve_case(data)

    data["geometry"]["fs_geom"]["l_micro"] = data["geometry"]["fs_geom"]["Lx_macro"]/data["geometry"]["fs_geom"]["nx_macro"] # length of a microscopic domain
    data["geometry"]["fs_geom"]["eps_ratio"] = data["geometry"]["fs_geom"]["l_micro"]/data["geometry"]["fs_geom"]["Lx_macro"] # multiscale parameters (ratio)

    # f_fun = make_load_function(data)
    # pb = PhunPeng(data, f_fun)
    # U, U_RStrain, U_Lagrange = pb.solve()

    # if data["flags"]["solve_hom"]:

    #     print("\nComputing effective coefficients...")
        
    #     cell_pb = CellProblem(
    #         data,
    #         output_dir="output/cellpb",
    #     )
    #     Ceff = cell_pb.solve_all()

    #     print(" done.")

    # if data["flags"]["solve_fs"]:
        
    #     print("\nSolving finescale problem...")
    #     f_fun = make_load_function(data)

    #     fs_pb = FineScaleProblem(data, f_fun)
    #     U_fs = fs_pb.solve()

    return 0


def solve_case(data):
    """
    Solve one complete fine-scale/effective problem.

    Returns
    -------
    U_eff : Function
    U_fs : Function
    """

    flags = data["flags"]
    dim = data["dim"]
    order = data["order"]

    
    data["geometry"]["fs_geom"]["l_micro"] = data["geometry"]["fs_geom"]["Lx_macro"]/data["geometry"]["fs_geom"]["nx_macro"] # length of a microscopic domain
    data["geometry"]["fs_geom"]["eps_ratio"] = data["geometry"]["fs_geom"]["l_micro"]/data["geometry"]["fs_geom"]["Lx_macro"] # multiscale parameters (ratio)

    # ------------------------------------------------------------
    # Material functions
    # ------------------------------------------------------------
    mu_fun, lmbda_fun = make_material_functions(data)
    Mu_fun, Lmbda_fun = make_material_functions(data, True)

    # ------------------------------------------------------------
    # Homogenization
    # ------------------------------------------------------------
    Ceff = None
    if flags["solve_hom"]:

        print("\nComputing effective coefficients...")

        cell_pb = CellProblem(
            data,
            output_dir="output/cellpb",
        )
        Ceff = cell_pb.solve_all()

        print(" done.")

    if flags["check_coeff"]:
        check_effective_coefficients(data, Ceff)

    f_fun = make_load_function(data)

    # ------------------------------------------------------------
    # Effective problem
    # ------------------------------------------------------------
    U_eff = None
    U_RStrain = None
    U_Lagrange = None

    if flags["solve_eff"]: 
        print("\nSolving effective problem...")
        if flags["ho_type"] == "linear":
            eff_pb = EffectiveProblemLinear(data, Ceff, f_fun)
            U_eff = eff_pb.solve()
        elif flags["ho_type"] == "strain_gradient":
            eff_pb = EffectiveProblemHo(data, Ceff, f_fun)
            U_eff, U_RStrain, U_Lagrange = eff_pb.solve()

        print(" done.")

    # ------------------------------------------------------------
    # Fine-scale problem
    # ------------------------------------------------------------
    U_fs = None
    if flags["solve_fs"]:
        
        print("\nSolving finescale problem...")

        fs_pb = FineScaleProblem(data, f_fun)
        U_fs = fs_pb.solve()

    if U_eff == None:
        return U_fs, None, None, None
    return U_fs, U_eff, U_RStrain, U_Lagrange


if __name__ == "__main__":
    main()



# solve_hom = True
# solve_eff = True
# solve_fs = True
# check_coeff = False
# is_per = True

# mirror_structure = True

# # DATA FOR THE LOCAL PROBLEM
# n = 50              # nref_cell
# data = {
#     "R": 5,
#     "t": 2.5,
#     "l": 25,
#     "mirror": mirror_structure,
#     "beta": np.pi/4
# }
# n_micro = 32        #nref_cell in fs_geom
# n_macro = 25        # total number of microscopic domain
# L_macro = 8               # length of macroscopic domain
# l_micro = L_macro/n_macro
# fs_geom = {
#     "nx_micro": n_micro,
#     "ny_micro": n_micro,
#     "nz_micro": 1,
#     "n_macro": n_macro,
#     "L_macro": L_macro,
#     "l_micro": L/nx
# }

# dim = 2
# order = 4
# loadx = 0
# loady = -5
# mu_dark = 1e10
# mu_light = 1e7
# lmbda_dark = 0
# lmbda_light = 0