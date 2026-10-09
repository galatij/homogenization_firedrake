from dataclasses import dataclass
import copy
import numpy as np
import os
from firedrake import *

from firedrake import (
    Function,
    VectorFunctionSpace,
    TensorFunctionSpace,
    interpolate,
    assemble,
    inner,
    grad,
    dx,
    sqrt,
)


@dataclass
class ConvergenceResult:
    nx_macro: int
    ny_macro: int
    
    epsilon_x: float
    epsilon_y: float
    epsilon: float

    avg_eff: float
    avg_fs: float

    avg_difference: float

    error_L2: float
    error_grad_L2: float

    rel_L2: float = None
    rel_grad_L2: float = None

    rate_L2: float = None
    rate_grad_L2: float = None


def spatial_average(U):
    """
    Compute the spatial average of a scalar or vector Function.
    """

    mesh = U.function_space().mesh()
    domain_measure = assemble(1.0 * dx(domain=mesh))

    if len(U.ufl_shape) == 0:
        return float(
            assemble(U * dx(domain=mesh)) / domain_measure
        )

    return np.array([
        assemble(U[i] * dx(domain=mesh)) / domain_measure
        for i in range(U.ufl_shape[0])
    ])

def convergence_rates(epsilons, errors):
    """
    Compute observed convergence rates for

        error ~ C * epsilon^p.

    For two consecutive values,

        p = log(E_i / E_{i+1})
            --------------------
            log(eps_i / eps_{i+1})

    """

    epsilons = np.asarray(epsilons, dtype=float)
    errors = np.asarray(errors, dtype=float)

    rates = np.full(len(errors), np.nan)

    for i in range(1, len(errors)):

        if (
            errors[i] > 0
            and errors[i - 1] > 0
            and epsilons[i] > 0
            and epsilons[i - 1] > 0
        ):

            rates[i] = (
                np.log(errors[i - 1] / errors[i])
                /
                np.log(epsilons[i - 1] / epsilons[i])
            )

    return rates

def compare_solutions(U_fs, U_eff, U_RStrain, U_Lagrange, eps, Lx, Ly, Lz=None, periodic=True):
    """
    Compare

        U_eff  = homogenized/effective solution
        U_fs   = fine-scale solution

    for the SAME epsilon.

    The fine-scale mesh is used as the common integration mesh.

    Returns
    -------
    dict
        Quantities needed by the homogenization convergence test.
    """

    nx_comp = max(1, int(round(Lx / eps)))
    ny_comp = max(1, int(round(Ly / eps)))

    if U_fs.function_space().mesh().comm.rank == 0:
        print(
            f"Comparison mesh: {nx_comp} x {ny_comp}, "
            f"h ~ ({Lx/nx_comp:.3e}, {Ly/ny_comp:.3e})"
        )

    if Lz is not None:
        nz_comp = max(1, int(round(Lz / eps)))

    if periodic:
        if Lz is None:
            mesh_comp = PeriodicRectangleMesh(8*nx_comp, 8*ny_comp, Lx, Ly)
        else:
            mesh_comp = PeriodicBoxMesh(8*nx_comp, 8*ny_comp, nz_comp, Lx, Ly, Lz)
    else:
        if Lz is None:
            mesh_comp = RectangleMesh(8*nx_comp, 8*ny_comp, Lx, Ly)
        else:
            mesh_comp = BoxMesh(8*nx_comp, 8*ny_comp, nz_comp, Lx, Ly, Lz)

    # V_comp = VectorFunctionSpace(
    #     U_eff.function_space().mesh(), "CG", U_eff.function_space().ufl_element().degree()
    # )

    V_comp = VectorFunctionSpace(
        mesh_comp, "CG", 2
    )

    # mesh_fs = U_fs.function_space().mesh()
    # V_fs = U_fs.function_space()

    # ------------------------------------------------------------
    # Interpolate effective displacement onto fine-scale mesh
    # ------------------------------------------------------------
    U_eff_comp = Function(V_comp)
    U_fs_comp = Function(V_comp)

    U_eff_comp.interpolate(U_eff)
    U_fs_comp.interpolate(U_fs)


    print("interpolation finished")

    # U_eff_fs = Function(V_fs)
    # U_eff_fs.interpolate(U_eff)

    # ------------------------------------------------------------
    # Averages
    # ------------------------------------------------------------

    avg_eff_vec = spatial_average(U_eff_comp)
    avg_fs_vec = spatial_average(U_fs_comp)

    print("spatial averages done")

    avg_eff = float(
        np.linalg.norm(avg_eff_vec)
    )

    avg_fs = float(
        np.linalg.norm(avg_fs_vec)
    )

    # Difference of spatial averages
    avg_difference = float(
        np.linalg.norm(
            avg_eff_vec - avg_fs_vec
        )
    )

    # ------------------------------------------------------------
    # L2 homogenization error
    #
    # || U_eff - U_fs ||_L2
    # ------------------------------------------------------------

    error_L2_sq = assemble(
        inner(
            U_eff_comp - U_fs_comp,
            U_eff_comp - U_fs_comp
        )
        * dx(domain=mesh_comp)
    )

    error_L2 = float(
        np.sqrt(max(float(error_L2_sq), 0.0))
    )

    print("error assembled")

    # ------------------------------------------------------------
    # Relative L2 error
    # ------------------------------------------------------------

    norm_fs_sq = assemble(
        inner(U_fs_comp, U_fs_comp)
        * dx(domain=mesh_comp)
    )

    norm_fs = float(
        np.sqrt(max(float(norm_fs_sq), 0.0))
    )

    norm_eff = sqrt(assemble(
        inner(U_eff_comp, U_eff_comp) * dx(domain=mesh_comp)
    ))

    rel_L2 = (
        error_L2 / norm_fs
        if norm_fs > 0
        else np.nan
    )

    # ------------------------------------------------------------
    # Gradient homogenization error
    #
    # || grad(U_eff) - grad(U_fs) ||_L2
    # ------------------------------------------------------------

    error_grad_sq = assemble(
        inner(
            grad(U_eff_comp) - grad(U_fs_comp),
            grad(U_eff_comp) - grad(U_fs_comp)
        )
        * dx(domain=mesh_comp)
    )

    error_grad_L2 = float(
        np.sqrt(max(float(error_grad_sq), 0.0))
    )

    # ------------------------------------------------------------
    # Relative gradient error
    # ------------------------------------------------------------

    norm_grad_fs_sq = assemble(
        inner(
            grad(U_fs_comp),
            grad(U_fs_comp)
        )
        * dx(domain=mesh_comp)
    )

    norm_grad_fs = float(
        np.sqrt(max(float(norm_grad_fs_sq), 0.0))
    )

    rel_grad_L2 = (
        error_grad_L2 / norm_grad_fs
        if norm_grad_fs > 0
        else np.nan
    )

    print("max |U_fs| :", np.max(np.abs(U_fs_comp.dat.data_ro)))
    print("max |U_eff|:", np.max(np.abs(U_eff_comp.dat.data_ro)))
    print("L2 |U_fs|  :", norm_fs)
    print("L2 |U_eff| :", norm_eff)
    print("L2 error   :", error_L2)
    print(
        f"||u_fs|| = {norm_fs:.6e}, "
        f"||u_eff|| = {norm_eff:.6e}, "
        f"error = {error_L2:.6e}, "
        f"rel_fs = {error_L2/norm_fs:.6e}, "
        f"rel_eff = {error_L2/norm_eff:.6e}"
    )

    return {
        "avg_eff": avg_eff,
        "avg_fs": avg_fs,
        "avg_difference": avg_difference,

        "error_L2": error_L2,
        "error_grad_L2": error_grad_L2,

        "rel_L2": rel_L2,
        "rel_grad_L2": rel_grad_L2,
    }

class HomogenizationConvergenceTest:
    """
    Convergence test for

        epsilon -> 0.

    For every epsilon, a NEW fine-scale problem is solved.

    The homogenized problem is compared with the corresponding
    fine-scale problem.

    Parameters
    ----------
    data : dict
        Base simulation dictionary.

    nx_macro_values : iterable
        Number of microdomains in x.

    ny_macro_values : iterable or None
        Number of microdomains in y.

    output_dir : str
        Directory for convergence results.
    """

    def __init__(
        self,
        data,
        nx_macro_values,
        ny_macro_values=None,
        output_dir="output/convergence",
    ):

        self.data = copy.deepcopy(data)

        self.nx_macro_values = list(
            nx_macro_values
        )

        if ny_macro_values is None:

            self.ny_macro_values = list(
                nx_macro_values
            )

        else:

            self.ny_macro_values = list(
                ny_macro_values
            )

        if len(self.nx_macro_values) != len(
            self.ny_macro_values
        ):

            raise ValueError(
                "nx_macro_values and ny_macro_values "
                "must have the same length."
            )

        self.output_dir = output_dir

        os.makedirs(
            self.output_dir,
            exist_ok=True
        )

        self.results = []


    # ============================================================
    # Compute epsilon
    # ============================================================

    @staticmethod
    def compute_epsilon(
        nx_macro,
        ny_macro,
        Lx_macro,
        Ly_macro,
    ):
        """
        Compute the dimensionless microstructural scale.

        epsilon_x = ell_x / Lx = 1 / nx_macro
        epsilon_y = ell_y / Ly = 1 / ny_macro

        For a rectangular problem we use

            epsilon = max(epsilon_x, epsilon_y)

        as the overall small parameter.
        """

        epsilon_x = 1.0 / nx_macro
        epsilon_y = 1.0 / ny_macro

        epsilon = max(
            epsilon_x,
            epsilon_y
        )

        return (
            epsilon_x,
            epsilon_y,
            epsilon,
        )


    # ============================================================
    # Run
    # ============================================================

    def run(self, solve_case):

        self.results = []

        Lx_macro = self.data["geometry"]["fs_geom"]["Lx_macro"]
        Ly_macro = self.data["geometry"]["fs_geom"]["Ly_macro"]

        for nx_macro, ny_macro in zip(
            self.nx_macro_values,
            self.ny_macro_values,
        ):

            print("\n")
            print("=" * 75)
            print("HOMOGENIZATION CONVERGENCE TEST")
            print("=" * 75)

            # ----------------------------------------------------
            # Create independent configuration
            # ----------------------------------------------------

            data_test = copy.deepcopy(self.data)

            fs_geom = data_test["geometry"]["fs_geom"]

            fs_geom["nx_macro"] = nx_macro

            fs_geom["ny_macro"] = ny_macro

            # ----------------------------------------------------
            # Compute epsilon
            # ----------------------------------------------------

            (epsilon_x, epsilon_y, epsilon) = self.compute_epsilon(
                nx_macro, ny_macro, Lx_macro, Ly_macro
                )

            print(f"nx_macro = {nx_macro}")
            print(f"ny_macro = {ny_macro}")

            print(f"epsilon_x = {epsilon_x:.8e}")

            print(f"epsilon_y = {epsilon_y:.8e}")

            print(f"epsilon   = {epsilon:.8e}")

            # ----------------------------------------------------
            # Solve the problem
            # ----------------------------------------------------

            U_fs, U_eff, U_RStrain, U_Lagrange = solve_case(data_test)

            # ----------------------------------------------------
            # Compare
            # ----------------------------------------------------

            errors = compare_solutions(
                U_fs, U_eff, U_RStrain, U_Lagrange, epsilon, Lx_macro, Ly_macro, Lz=None, periodic=True
            )

            # ----------------------------------------------------
            # Store result
            # ----------------------------------------------------

            result = ConvergenceResult(
                nx_macro=nx_macro,
                ny_macro=ny_macro,

                epsilon_x=epsilon_x,
                epsilon_y=epsilon_y,
                epsilon=epsilon,

                **errors,
            )

            self.results.append(result)

            # ----------------------------------------------------
            # Print
            # ----------------------------------------------------

            print()
            print(f"|<U_eff>|        = {result.avg_eff:.8e}")
            print(f"|<U_fs>|         = {result.avg_fs:.8e}")

            print(f"|<U_eff>-<U_fs>| = {result.avg_difference:.8e}")
            print(f"L2 error         = {result.error_L2:.8e}")
            print(f"L2 relative      = {result.rel_L2:.8e}")
            print(f"grad L2 error    = {result.error_grad_L2:.8e}")
            print(f"grad relative    = {result.rel_grad_L2:.8e}")

            

        # --------------------------------------------------------
        # Compute convergence rates AFTER all cases
        # --------------------------------------------------------

        self.compute_rates()

        return self.results


    # ============================================================
    # Rates
    # ============================================================

    def compute_rates(self):

        epsilons = np.array([
            r.epsilon
            for r in self.results
        ])

        errors_L2 = np.array([
            r.error_L2
            for r in self.results
        ])

        errors_grad = np.array([
            r.error_grad_L2
            for r in self.results
        ])

        rates_L2 = convergence_rates(
            epsilons,
            errors_L2
        )

        rates_grad = convergence_rates(
            epsilons,
            errors_grad
        )

        for i, result in enumerate(
            self.results
        ):

            result.rate_L2 = rates_L2[i]

            result.rate_grad_L2 = rates_grad[i]


    # ============================================================
    # Print convergence table
    # ============================================================
    def print_table(self):

        microstructure = self.data["flags"]["microstructure"]

        mu_light = self.data["coefficients"]["mu_light"]
        mu_dark = self.data["coefficients"]["mu_dark"]
        t_ratio = self.data["geometry"]["crossed"]["t"]

        filename = f"output/{microstructure}_convergence.txt"

        # Header for this simulation
        if mu_dark == mu_light:
            material_case = " (HOMOGENEOUS CASE)"
        else:
            material_case = ""

        lines = []

        lines.append("")
        lines.append(f'Microstructure: "{microstructure}"')
        lines.append(f"mu_light: {mu_light:.0e}")
        lines.append(f"mu_dark: {mu_dark:.0e}{material_case}")
        lines.append(f"t_ratio: {t_ratio}")
        lines.append("-" * 95)

        lines.append(
            f"{'epsilon':>12}"
            f"{'nx':>8}"
            f"{'L2 error':>16}"
            f"{'L2 rate':>12}"
            f"{'grad error':>16}"
            f"{'grad rate':>12}"
        )

        lines.append("-" * 95)

        for r in self.results:

            lines.append(
                f"{r.epsilon:12.4e}"
                f"{r.nx_macro:8d}"
                f"{r.error_L2:16.6e}"
                f"{r.rate_L2:12.4f}"
                f"{r.error_grad_L2:16.6e}"
                f"{r.rate_grad_L2:12.4f}"
            )

        lines.append("-" * 95)

        # Print to terminal
        print()
        for line in lines:
            print(line)

        # Append to convergence file
        with open(filename, "a") as f:
            f.write("\n".join(lines))
            f.write("\n")
    # def print_table(self):

    #     print()
    #     print(
    #         "Observed homogenization convergence rates:"
    #     )

    #     print("-" * 95)

    #     print(
    #         f"{'epsilon':>12}"
    #         f"{'nx':>8}"
    #         f"{'L2 error':>16}"
    #         f"{'L2 rate':>12}"
    #         f"{'grad error':>16}"
    #         f"{'grad rate':>12}"
    #     )

    #     print("-" * 95)

    #     for r in self.results:

    #         print(
    #             f"{r.epsilon:12.4e}"
    #             f"{r.nx_macro:8d}"
    #             f"{r.error_L2:16.6e}"
    #             f"{r.rate_L2:12.4f}"
    #             f"{r.error_grad_L2:16.6e}"
    #             f"{r.rate_grad_L2:12.4f}"
    #         )

    #     print("-" * 95)


    # ============================================================
    # Plot errors
    # ============================================================

    def plot_convergence(
        self,
        save=True,
        show=True,
    ):

        import matplotlib.pyplot as plt

        epsilon = np.array([
            r.epsilon
            for r in self.results
        ])

        error_L2 = np.array([
            r.error_L2
            for r in self.results
        ])

        error_grad = np.array([
            r.error_grad_L2
            for r in self.results
        ])

        # --------------------------------------------------------
        # L2 displacement error
        # --------------------------------------------------------

        fig = plt.figure()

        plt.loglog(
            epsilon,
            error_L2,
            "o-",
            label=r"$\|U^\varepsilon-U^0\|_{L^2}$"
        )

        plt.xlabel(
            r"$\varepsilon$"
        )

        plt.ylabel(
            "L2 error"
        )

        plt.grid(
            True,
            which="both"
        )

        plt.legend()

        plt.gca().invert_xaxis()

        plt.tight_layout()

        if save:

            filename = os.path.join(
                self.output_dir,
                "homogenization_L2.png"
            )

            plt.savefig(
                filename,
                dpi=300,
                bbox_inches="tight"
            )

            print(
                f"Saved: {filename}"
            )

        if show:
            plt.show()

        # --------------------------------------------------------
        # Gradient error
        # --------------------------------------------------------

        fig = plt.figure()

        plt.loglog(
            epsilon,
            error_grad,
            "s-",
            label=(
                r"$\|\nabla U^\varepsilon"
                r"-\nabla U^0\|_{L^2}$"
            )
        )

        plt.xlabel(
            r"$\varepsilon$"
        )

        plt.ylabel(
            "Gradient L2 error"
        )

        plt.grid(
            True,
            which="both"
        )

        plt.legend()

        plt.gca().invert_xaxis()

        plt.tight_layout()

        if save:

            filename = os.path.join(
                self.output_dir,
                "homogenization_gradient_L2.png"
            )

            plt.savefig(
                filename,
                dpi=300,
                bbox_inches="tight"
            )

            print(
                f"Saved: {filename}"
            )

        if show:
            plt.show()

        return fig


    # ============================================================
    # Plot relative errors
    # ============================================================

    def plot_relative_errors(
        self,
        save=True,
        show=True,
    ):

        import matplotlib.pyplot as plt

        epsilon = np.array([
            r.epsilon
            for r in self.results
        ])

        error_L2 = np.array([
            r.rel_L2
            for r in self.results
        ])

        error_grad = np.array([
            r.rel_grad_L2
            for r in self.results
        ])

        plt.figure()

        plt.loglog(
            epsilon,
            error_L2,
            "o-",
            label=r"$\|U^\varepsilon-U^0\|/\|U^\varepsilon\|$"
        )

        plt.loglog(
            epsilon,
            error_grad,
            "s-",
            label=(
                r"$\|\nabla U^\varepsilon"
                r"-\nabla U^0\|/"
                r"\|\nabla U^\varepsilon\|$"
            )
        )

        plt.xlabel(
            r"$\varepsilon$"
        )

        plt.ylabel(
            "relative error"
        )

        plt.grid(
            True,
            which="both"
        )

        plt.legend()

        plt.gca().invert_xaxis()

        plt.tight_layout()

        if save:

            filename = os.path.join(
                self.output_dir,
                "homogenization_relative.png"
            )

            plt.savefig(
                filename,
                dpi=300,
                bbox_inches="tight"
            )

            print(
                f"Saved: {filename}"
            )

        if show:
            plt.show()


# ================================================================
# Generate epsilon sequence
# ================================================================

def homogenization_sequence(
    n_start,
    n_end,
):
    """
    Generate

        n_start, 2*n_start, 4*n_start, ...

    corresponding to

        epsilon, epsilon/2, epsilon/4, ...

    when Lx_macro and Ly_macro are fixed.
    """

    values = [n_start]

    while values[-1] < n_end:

        values.append(
            2 * values[-1]
        )

    if values[-1] != n_end:

        raise ValueError(
            f"{n_end} cannot be reached from "
            f"{n_start} by repeated doubling."
        )

    return values