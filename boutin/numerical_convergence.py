import copy
import numpy as np
import matplotlib.pyplot as plt

from firedrake import *


def solution_difference_errors(U_coarse, U_fine):
    """
    Compare two numerical solutions on different meshes.

    The coarse solution is interpolated onto the fine mesh.

    Returns
    -------
    dict
        L2 and H1 errors and relative errors.
    """

    mesh_fine = U_fine.function_space().mesh()
    V_fine = U_fine.function_space()

    # Transfer coarse solution to fine space
    U_coarse_fine = Function(V_fine)
    U_coarse_fine.interpolate(U_coarse)

    # Difference
    e = U_coarse_fine - U_fine

    # --------------------------------------------------
    # L2 error
    # --------------------------------------------------

    error_L2_sq = assemble(
        inner(e, e) * dx(domain=mesh_fine)
    )

    error_L2 = np.sqrt(
        max(float(error_L2_sq), 0.0)
    )

    # --------------------------------------------------
    # H1 seminorm
    # --------------------------------------------------

    error_H1_semi_sq = assemble(
        inner(grad(e), grad(e))
        * dx(domain=mesh_fine)
    )

    error_H1_semi = np.sqrt(
        max(float(error_H1_semi_sq), 0.0)
    )

    # --------------------------------------------------
    # Full H1 norm
    # --------------------------------------------------

    error_H1 = np.sqrt(
        error_L2**2 + error_H1_semi**2
    )

    # --------------------------------------------------
    # Norm of fine solution
    # --------------------------------------------------

    norm_L2_sq = assemble(
        inner(U_fine, U_fine)
        * dx(domain=mesh_fine)
    )

    norm_H1_semi_sq = assemble(
        inner(grad(U_fine), grad(U_fine))
        * dx(domain=mesh_fine)
    )

    norm_L2 = np.sqrt(
        max(float(norm_L2_sq), 0.0)
    )

    norm_H1 = np.sqrt(
        float(norm_L2_sq) +
        float(norm_H1_semi_sq)
    )

    norm_Linf = np.max(np.abs(U_fine.dat.data_ro))

    # --------------------------------------------------
    # Relative errors
    # --------------------------------------------------

    relative_L2 = (
        error_L2 / norm_L2
        if norm_L2 > 0
        else np.nan
    )

    relative_H1 = (
        error_H1 / norm_H1
        if norm_H1 > 0
        else np.nan
    )

    return {
        "L2": error_L2,
        "H1": error_H1,
        "H1_semi": error_H1_semi,
        "rel_L2": relative_L2,
        "rel_H1": relative_H1,
        "||U||_0": norm_L2,
        "||U||_inf": norm_Linf,
    }


def convergence_rates_fem(errors, h_values):
    """
    Compute empirical convergence rates.

    rate_i =
        log(error_i-1 / error_i)
        -----------------------
        log(h_i-1 / h_i)
    """

    rates = [np.nan]

    for i in range(1, len(errors)):

        rate = (
            np.log(errors[i-1] / errors[i])
            /
            np.log(h_values[i-1] / h_values[i])
        )

        rates.append(rate)

    return rates


def run_fem_convergence(
    data,
    solve_case,
    plot=True,
):
    """
    FEM self-convergence study for the fine-scale and/or
    effective problem.

    The problems that are actually solved are controlled by

        data["flags"]["solve_fs"]
        data["flags"]["solve_eff"]

    Parameters
    ----------
    data : dict
        Base input dictionary.

    plot : bool
        If True, plot errors and convergence rates.

    Returns
    -------
    results : dict
        Contains results for every enabled problem.
    """

    solve_fs = data["flags"]["solve_fs"]
    solve_eff = data["flags"]["solve_eff"]
    num_refinement = data["convergence"]["numerical_refinements"] # NUMBER OF REFINEMENTS: 2^0, 2^1, ... 2^(num_refinement-1)

    if not solve_fs and not solve_eff:
        raise ValueError(
            "Both solve_fs and solve_eff are False."
        )

    results = {}

    # ============================================================
    # Fine-scale FEM convergence
    # ============================================================

    if solve_fs:

        print("\n")
        print("=" * 120)
        print("FINE-SCALE FEM CONVERGENCE")
        print("=" * 120)

        solutions_fs = []
        h_fs = []
        refinement_fs = []
        for ref_level in range(0,num_refinement):
            refinement_fs.append(ref_level)
            print("\n" + "-" * 100)
            print(f"Fine-scale: ref_level = {ref_level}")
            print("-" * 100)
            
            n_micro = 2**(ref_level)

            data_i = copy.deepcopy(data)

            data_i["convergence"]["current_level"] = ref_level
            data_i["geometry"]["fs_geom"]["n_micro_fs"] = n_micro

            # Only solve the fine-scale problem
            data_i["flags"]["solve_fs"] = True
            data_i["flags"]["solve_eff"] = False

            U_fs, _, _, _ = solve_case(data_i)

            if U_fs is None:
                raise RuntimeError(
                    f"Fine-scale solution is None for "
                    f"ref_level = {ref_level}"
                )

            solutions_fs.append(U_fs)

            # ----------------------------------------------------
            # Physical FEM mesh size
            #
            # n_micro_fs = number of FEM cells per microstructure
            #
            # l_micro = Lx_macro / nx_macro
            # h_fs    = l_micro / n_micro_fs
            # ----------------------------------------------------

            geom = data_i["geometry"]["fs_geom"]

            Lx = geom["Lx_macro"]
            nx_macro = geom["nx_macro"]

            l_micro = Lx / nx_macro

            h = l_micro / n_micro

            h_fs.append(h)

        # --------------------------------------------------------
        # Self-convergence errors
        # --------------------------------------------------------

        errors_L2 = []
        errors_H1 = []

        rel_L2 = []
        rel_H1 = []

        norm_L2 = []
        norm_Linf = []

        for i in range(len(solutions_fs) - 1):

            print(
                f"\nComparing fine-scale solutions "
                f"{refinement_fs[i]} -> "
                f"{refinement_fs[i+1]}"
            )

            err = solution_difference_errors(
                solutions_fs[i],
                solutions_fs[i+1],
            )

            errors_L2.append(err["L2"])
            errors_H1.append(err["H1"])

            rel_L2.append(err["rel_L2"])
            rel_H1.append(err["rel_H1"])

            norm_L2.append(err["||U||_0"])
            norm_Linf.append(err["||U||_inf"])

        # --------------------------------------------------------
        # Rates
        #
        # errors correspond to intervals
        # [h_i, h_{i+1}]
        # --------------------------------------------------------

        rate_L2 = convergence_rates_fem(
            errors_L2,
            h_fs[:-1],
        )

        rate_H1 = convergence_rates_fem(
            errors_H1,
            h_fs[:-1],
        )

        results["finescale"] = {
            "refinement": refinement_fs,
            "h": h_fs,
            "solutions": solutions_fs,

            "L2": errors_L2,
            "H1": errors_H1,

            "rel_L2": rel_L2,
            "rel_H1": rel_H1,

            "rate_L2": rate_L2,
            "rate_H1": rate_H1,

            "norm_L2": norm_L2,
            "norm_Linf": norm_Linf
        }

        # --------------------------------------------------------
        # Print table
        # --------------------------------------------------------

        print("\n")
        print("=" * 120)
        print("FINE-SCALE FEM CONVERGENCE RESULTS")
        print("=" * 120)

        print(
            f"{'n_micro':>10}"
            f"{'h':>14}"
            f"{'L2 error':>18}"
            f"{'L2 rate':>12}"
            f"{'H1 error':>18}"
            f"{'H1 rate':>12}"
            f"{'L2 norm':>18}"
            f"{'Linf norm':>18}"
        )

        print("-" * 100)

        for i in range(len(errors_L2)):

            print(
                f"{refinement_fs[i]:10d}"
                f"{h_fs[i]:14.4e}"
                f"{errors_L2[i]:18.6e}"
                f"{rate_L2[i]:12.3f}"
                f"{errors_H1[i]:18.6e}"
                f"{rate_H1[i]:12.3f}"
                f"{norm_L2[i]:18.6e}"
                f"{norm_Linf[i]:18.6e}"
            )

        # Last refinement has no comparison
        i = len(refinement_fs) - 1

        print(
            f"{refinement_fs[i]:10d}"
            f"{h_fs[i]:14.4e}"
            f"{'---':>18}"
            f"{'---':>12}"
            f"{'---':>18}"
            f"{'---':>12}"
            f"{'---':>18}"
            f"{'---':>18}"
        )



    # ============================================================
    # Effective FEM convergence
    # ============================================================

    if solve_eff:

        print("\n")
        print("=" * 120)
        print("EFFECTIVE FEM CONVERGENCE")
        print("=" * 120)

        solutions_eff = []
        h_eff = []
        refinement_eff = []

        for ref_level in range(0,num_refinement):
            refinement_eff.append(ref_level)
            print("\n" + "-" * 100)
            print(f"Effective: ref_level = {ref_level}")
            print("-" * 100)
            
            n_micro = 2**(ref_level)

            data_i = copy.deepcopy(data)
            
            data_i["convergence"]["current_level"] = ref_level
            data_i["geometry"]["fs_geom"]["n_micro_eff"] = n_micro

            # Only solve the effective problem
            data_i["flags"]["solve_fs"] = False
            data_i["flags"]["solve_eff"] = True

            U_fs, U_eff, _, _ = solve_case(data_i)

            if U_eff is None:
                raise RuntimeError(
                    f"Effective solution is None for "
                    f"ref_level = {ref_level}"
                )

            solutions_eff.append(U_eff)

            # ----------------------------------------------------
            # Effective mesh size
            #
            # Assuming n_micro_eff is the number of effective
            # FEM cells per macroscopic length.
            # ----------------------------------------------------

            geom = data_i["geometry"]["fs_geom"]

            Lx = geom["Lx_macro"]

            h = Lx / n_micro

            h_eff.append(h)

        # --------------------------------------------------------
        # Self-convergence errors
        # --------------------------------------------------------

        errors_L2 = []
        errors_H1 = []

        rel_L2 = []
        rel_H1 = []

        norm_L2 = []
        norm_Linf = []

        for i in range(len(solutions_eff) - 1):

            print(
                f"\nComparing effective solutions "
                f"{refinement_eff[i]} -> "
                f"{refinement_eff[i+1]}"
            )

            err = solution_difference_errors(
                solutions_eff[i],
                solutions_eff[i+1],
            )

            errors_L2.append(err["L2"])
            errors_H1.append(err["H1"])

            rel_L2.append(err["rel_L2"])
            rel_H1.append(err["rel_H1"])

            norm_L2.append(err["||U||_0"])
            norm_Linf.append(err["||U||_inf"])

        # --------------------------------------------------------
        # Rates
        # --------------------------------------------------------

        rate_L2 = convergence_rates_fem(
            errors_L2,
            h_eff[:-1],
        )

        rate_H1 = convergence_rates_fem(
            errors_H1,
            h_eff[:-1],
        )

        results["effective"] = {
            "refinement": refinement_eff,
            "h": h_eff,
            "solutions": solutions_eff,

            "L2": errors_L2,
            "H1": errors_H1,

            "rel_L2": rel_L2,
            "rel_H1": rel_H1,

            "rate_L2": rate_L2,
            "rate_H1": rate_H1,


        }

        # --------------------------------------------------------
        # Print table
        # --------------------------------------------------------

        print("\n")
        print("=" * 120)
        print("EFFECTIVE FEM CONVERGENCE RESULTS")
        print("=" * 120)

        print(
            f"{'n_micro':>10}"
            f"{'h':>14}"
            f"{'L2 error':>18}"
            f"{'L2 rate':>12}"
            f"{'H1 error':>18}"
            f"{'H1 rate':>12}"
            f"{'L2 norm':>18}"
            f"{'Linf norm':>18}"
        )

        print("-" * 100)

        for i in range(len(errors_L2)):

            print(
                f"{refinement_eff[i]:10d}"
                f"{h_eff[i]:14.4e}"
                f"{errors_L2[i]:18.6e}"
                f"{rate_L2[i]:12.3f}"
                f"{errors_H1[i]:18.6e}"
                f"{rate_H1[i]:12.3f}"
                f"{norm_L2[i]:18.6e}"
                f"{norm_Linf[i]:18.6e}"
            )

        i = len(refinement_eff) - 1

        print(
            f"{refinement_eff[i]:10d}"
            f"{h_eff[i]:14.4e}"
            f"{'---':>18}"
            f"{'---':>12}"
            f"{'---':>18}"
            f"{'---':>12}"
            f"{'---':>18}"
            f"{'---':>18}"
        )

    # ============================================================
    # Plots
    # ============================================================

    if plot:

        # --------------------------------------------------------
        # Fine-scale plots
        # --------------------------------------------------------

        if solve_fs:

            r = results["finescale"]

            plt.figure()

            plt.loglog(
                r["h"][:-1],
                r["L2"],
                "o-",
                label=r"$L^2$",
            )

            plt.loglog(
                r["h"][:-1],
                r["H1"],
                "s-",
                label=r"$H^1$",
            )

            plt.xlabel(r"$h$")
            plt.ylabel("successive-solution error")
            plt.title("Fine-scale FEM convergence")

            plt.grid(True, which="both")
            plt.legend()
            plt.gca().invert_xaxis()

            plt.show()

            plt.figure()

            plt.semilogx(
                r["h"][:-1],
                r["rate_L2"],
                "o-",
                label=r"$L^2$ rate",
            )

            plt.semilogx(
                r["h"][:-1],
                r["rate_H1"],
                "s-",
                label=r"$H^1$ rate",
            )

            plt.xlabel(r"$h$")
            plt.ylabel("observed rate")
            plt.title("Fine-scale FEM convergence rates")

            plt.grid(True, which="both")
            plt.legend()
            plt.gca().invert_xaxis()

            plt.show()

        # --------------------------------------------------------
        # Effective plots
        # --------------------------------------------------------

        if solve_eff:

            r = results["effective"]

            plt.figure()

            plt.loglog(
                r["h"][:-1],
                r["L2"],
                "o-",
                label=r"$L^2$",
            )

            plt.loglog(
                r["h"][:-1],
                r["H1"],
                "s-",
                label=r"$H^1$",
            )

            plt.xlabel(r"$h$")
            plt.ylabel("successive-solution error")
            plt.title("Effective FEM convergence")

            plt.grid(True, which="both")
            plt.legend()
            plt.gca().invert_xaxis()

            plt.show()

            plt.figure()

            plt.semilogx(
                r["h"][:-1],
                r["rate_L2"],
                "o-",
                label=r"$L^2$ rate",
            )

            plt.semilogx(
                r["h"][:-1],
                r["rate_H1"],
                "s-",
                label=r"$H^1$ rate",
            )

            plt.xlabel(r"$h$")
            plt.ylabel("observed rate")
            plt.title("Effective FEM convergence rates")

            plt.grid(True, which="both")
            plt.legend()
            plt.gca().invert_xaxis()

            plt.show()

    return results