from firedrake import *
from ufl import as_vector, as_matrix
import numpy as np
import os
from firedrake.output import VTKFile


class CellProblem:

    def __init__(self, n, mu, lmbda, dim, order, output_dir="output/primal"):

        print("Initializing primal cell problem...")

        self.dim = dim
        self.mu_fun = mu
        self.lmbda_fun = lmbda
        self.beta = 1
        self.order = order
        self.output_dir = output_dir

        # Mesh
        if dim == 3:
            self.mesh = PeriodicUnitCubeMesh(n, n, n, hexahedral=True)
        elif dim == 2:
            self.mesh = PeriodicUnitSquareMesh(n, n, quadrilateral=True)
        else:
            raise ValueError("dim must be 2 or 3")

        if order not in [2, 4]:
            raise ValueError(
                "order must be 2 or 4"
            )
        
        # Function spaces
        self.gradFEspace = TensorFunctionSpace(self.mesh, "DG", 0)
        self.P0 = FunctionSpace(self.mesh, "DG", 0)
        self.P1 = VectorFunctionSpace(self.mesh, "CG", 1)

        # Material coefficients
        if dim == 3:
            x, y, z = SpatialCoordinate(self.mesh)
        else:
            x, y = SpatialCoordinate(self.mesh)
            z = Constant(0.0)

        mu_expr = self.mu_fun(x, y, z)
        lmbda_expr = self.lmbda_fun(x, y, z)

        self.mu = Function(self.P0).interpolate(mu_expr)
        self.lmbda = Function(self.P0).interpolate(lmbda_expr)

        self.mu.rename("mu")
        self.lmbda.rename("lambda")

        os.makedirs(output_dir, exist_ok=True)

        VTKFile(os.path.join(output_dir, "mu.pvd")).write(self.mu)
        VTKFile(os.path.join(output_dir, "lambda.pvd")).write(self.lmbda)

        i, j, r, t = indices(4)
        I = Identity(self.dim)

        self.cst = as_tensor(
            self.mu * (
                I[i,r] * I[j,t]
                +
                I[i,t] * I[j,r]
            )
            +
            self.lmbda * I[i, j] * I[r, t],
            (i, j, r, t)
        )

        # Nullspace
        basis = []

        for k in range(self.dim):
            v = Function(self.P1)
            value = np.zeros(self.dim)
            value[k] = 1.0
            v.interpolate(Constant(tuple(value)))
            basis.append(v)
        self.nullspace = VectorSpaceBasis(basis)
        self.nullspace.orthonormalize()

        # Variational problem
        u = TrialFunction(self.P1)
        v = TestFunction(self.P1)

        eps_u = self.symgrad(u)

        i, j, r, t = indices(4)
        sigma = as_tensor(
            sum(
                self.cst[i, j, r, t] * eps_u[r, t]
                for r in range(self.dim)
                for t in range(self.dim)
            ),
            (i, j)
        )

        self.a = inner(sigma, self.symgrad(v)) * dx

        # Assemble
        print("Assembling...")

        A = assemble(
            self.a,
            mat_type="aij"
        )

        print(" done.")

        self.solver = LinearSolver(
            A,
            nullspace=self.nullspace,
            solver_parameters={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps"
            }
        )

        # --------------------------------------------------
        # Storage
        # --------------------------------------------------

        self.X = {}
        self.Y = {}
        self.Z = {}

        self.c0 = {}
        self.c1 = {}
        self.c2 = {}

        self.C0 = {}
        self.C1 = {}
        self.C2 = {}

        print("Initialization done.")


    def symgrad(self, u):

        return 0.5 * (
            grad(u) + grad(u).T
        )


    def solve_corrector(
        self,
        rhs=None,
        known_stress=None,
        name="corrector"
    ):

        v = TestFunction(self.P1)
        L = 0
        if rhs is not None:
            L += dot(rhs, v) *dx

        # --------------------------------------------------
        # Known stress contribution
        # --------------------------------------------------

        if known_stress is not None:
            L -= inner(known_stress, self.symgrad(v)) * dx
        b = assemble(L)

        w = Function(self.P1)
        self.solver.solve(w, b)
        w.rename(name)

        return w

    def average_tensor(self, tensor):

        volume = assemble(Constant(1.0) * dx(domain=self.mesh))
        shape = tensor.ufl_shape

        result = np.zeros(shape)

        for idx in np.ndindex(shape):
            result[idx] = (
                assemble(tensor[idx] * dx(domain=self.mesh)) / volume
            )

        return result

    def tensor_to_dg0(
        self,
        expr,
        name
    ):

        tensor = Function(self.gradFEspace, name=name)
        tensor.interpolate(expr)

        return tensor

    def solve_order0(self, k, l):

        print(f"\nSolving order-0 cell problem ({k},{l})")

        # --------------------------------------------------
        # A_macro = e_k \otimes e_l
        # --------------------------------------------------

        e_k = np.zeros(self.dim)
        e_l = np.zeros(self.dim)

        e_k[k] = 1.0
        e_l[l] = 1.0

        e_k = as_vector(tuple(e_k))
        e_l = as_vector(tuple(e_l))

        A_macro = outer(e_k, e_l)

        # --------------------------------------------------
        # c : A_macro
        # --------------------------------------------------

        i, j, r, t = indices(4)

        sigma_macro = as_tensor(
            sum(
                self.cst[i, j, r, t]
                * A_macro[r, t]
                for r in range(self.dim)
                for t in range(self.dim)
            ),
            (i, j)
        )

        # --------------------------------------------------
        # Solve
        #
        # -div(c:eps(X)) = div(c:A_macro)
        #
        # equivalently
        #
        # a(X,v) = - integral c:A_macro:eps(v)
        # --------------------------------------------------

        X = self.solve_corrector(
            rhs=None,
            known_stress=sigma_macro,
            name=f"X_{k}{l}"
        )

        self.X[(k, l)] = X

        # --------------------------------------------------
        # Microscopic strain
        # --------------------------------------------------

        eps_X = 0.5 * (A_macro + A_macro.T) + self.symgrad(X)

        # --------------------------------------------------
        # c0_ij^{kl}
        # --------------------------------------------------

        c0_expr = as_tensor(
            sum(
                self.cst[i, j, r, t]
                * eps_X[r, t]
                for r in range(self.dim)
                for t in range(self.dim)
            ),
            (i, j)
        )

        c0 = self.tensor_to_dg0(
            c0_expr,
            f"c0_{k}{l}"
        )

        self.c0[(k, l)] = c0

        C0 = self.average_tensor(c0)
        self.C0[(k, l)] = C0

        print("||X|| =", norm(X))

        print("||eps|| =", norm(eps_X))

        return X, c0, C0


    def solve_order1(self, k, l, m):

        print(f"\nSolving order-1 cell problem ({k},{l},{m})")

        # --------------------------------------------------
        # Previous solution X^{kl}
        # --------------------------------------------------
        X = self.X[(k,l)]

        # --------------------------------------------------
        # Previous coefficient c0^{kl}
        # --------------------------------------------------
        c0 = self.c0[(k, l)]
        C0 = self.C0[(k, l)]

        # --------------------------------------------------
        # Known stress:
        #
        # sigma_known_ij =
        # c_ij^{mr} X_r^{kl}
        # --------------------------------------------------

        i, j, r = indices(3)

        known_stress = as_tensor(
            sum(
                self.cst[i, j, m, r] * X[r]
                for r in range(self.dim)
            ),
            (i, j)
        )

        # --------------------------------------------------
        # R_i =
        # beta C0_im^{kl}
        # - c0_im^{kl}
        #
        # Our solver wants F = -R.
        # --------------------------------------------------

        rhs = as_vector([
            c0[p, m]
            - self.beta * C0[p, m]
            for p in range(self.dim)
        ])

        # --------------------------------------------------
        # Solve for Y
        # --------------------------------------------------

        Y = self.solve_corrector(
            rhs=rhs,
            known_stress=known_stress,
            name=f"Y_{k}{l}{m}"
        )

        self.Y[(k, l, m)] = Y

        # --------------------------------------------------
        # c1^{klm}
        # --------------------------------------------------

        eps_Y = self.symgrad(Y)

        c1_expr = known_stress + as_tensor(
            sum(
                self.cst[i, j, r, t]
                * eps_Y[r, t]
                for r in range(self.dim)
                for t in range(self.dim)
            ),
            (i, j)
        )

        c1 = self.tensor_to_dg0(
            c1_expr,
            f"c1_{k}{l}{m}"
        )

        self.c1[(k, l, m)] = c1

        C1 = self.average_tensor(c1)
        self.C1[(k, l, m)] = C1

        print("||Y|| =", norm(Y))

        return Y, c1, C1


    def solve_order2(
        self,
        k,
        l,
        m,
        n
    ):

        print(f"\nSolving order-2 cell problem ({k},{l},{m},{n})")

        # --------------------------------------------------
        # Previous solution Y^{klm}
        # --------------------------------------------------
        Y = self.Y[(k, l, m)]

        # --------------------------------------------------
        # Previous coefficient c1^{klm}
        # --------------------------------------------------
        c1 = self.c1[(k, l, m)]
        C1 = self.C1[(k, l, m)]

        # --------------------------------------------------
        # Known stress:
        #
        # sigma_known_ij =
        # c_ij^{nr} Y_r^{klm}
        # --------------------------------------------------

        i, j, r, s, t = indices(5)

        known_stress = as_tensor(
            sum(
                self.cst[i, j, n, r] * Y[r]
                for r in range(self.dim)
            ),
            (i, j)
        )

        # --------------------------------------------------
        # R_i =
        # beta C1_in^{klm}
        # - c1_in^{klm}
        #
        # Solver wants F = -R.
        # --------------------------------------------------

        rhs = as_vector([
            c1[p, n] - self.beta * C1[p, n]
            for p in range(self.dim)
        ])

        # --------------------------------------------------
        # Solve Z
        # --------------------------------------------------

        Z = self.solve_corrector(
            rhs=rhs,
            known_stress=known_stress,
            name=f"Z_{k}{l}{m}{n}"
        )

        self.Z[(k, l, m, n)] = Z

        # --------------------------------------------------
        # c2
        # --------------------------------------------------
        eps_Z = self.symgrad(Z)

        c2_expr = (
            known_stress
            +
            as_tensor(
                sum(
                    self.cst[i, j, s, t] * eps_Z[s, t]
                    for s in range(self.dim)
                    for t in range(self.dim)
                ),
                (i, j)
            )
        )

        c2 = self.tensor_to_dg0(
            c2_expr,
            f"c2_{k}{l}{m}{n}"
        )

        self.c2[(k, l, m, n)] = c2
        C2 = self.average_tensor(c2)
        self.C2[(k, l, m, n)] = C2

        print("||Z|| =", norm(Z))

        return Z, c2, C2


    def solve(self, order):

        if order == 0:

            for k in range(self.dim):
                for l in range(self.dim):
                    self.solve_order0(k, l)

        elif order == 1:
            for k in range(self.dim):
                for l in range(self.dim):
                    for m in range(self.dim):
                        self.solve_order1(k, l, m)

        elif order == 2:
            for k in range(self.dim):
                for l in range(self.dim):
                    for m in range(self.dim):
                        for n in range(self.dim):
                            self.solve_order2(k, l, m, n)

        else:
            raise ValueError(
                "Only orders 0, 1 and 2 are implemented."
            )

    def solve_all(self):

        # --------------------------------------------------
        # Always solve order 0
        # --------------------------------------------------
        self.solve(0)

        # --------------------------------------------------
        # Order 1
        # --------------------------------------------------
        if self.order >= 4:

            self.solve(1)
            self.solve(2)

        self._save_coefficients()

        return {
            0: self.C0,
            1: self.C1,
            2: self.C2
        }

    def _save_coefficients(self):

        folder = self.output_dir

        np.save(
            os.path.join(
                folder,
                "C0.npy"
            ),
            self.C0,
            allow_pickle=True
        )

        if len(self.C1) > 0:
            np.save(
                os.path.join(
                    folder,
                    "C1.npy"
                ),
                self.C1,
                allow_pickle=True
            )

        if len(self.C2) > 0:
            np.save(
                os.path.join(
                    folder,
                    "C2.npy"
                ),
                self.C2,
                allow_pickle=True
            )

    # def solve(self, l, m):

    #     """
    #     Solve

    #         -div [ c : (e_l x e_m + eps(X)) ] = 0

    #     for the displacement corrector X^{lm}.
    #     """

    #     print(
    #         f"\nSolving primal cell problem ({l},{m})"
    #     )

    #     # --------------------------------------------------
    #     # Basis tensor A = e_l \otimes e_m
    #     # --------------------------------------------------
    #     e_l = np.zeros(self.dim)
    #     e_m = np.zeros(self.dim)

    #     e_l[l] = 1.0
    #     e_m[m] = 1.0

    #     e_l = as_vector(tuple(e_l))
    #     e_m = as_vector(tuple(e_m))

    #     A_macro = outer(e_l, e_m)
    #     # A_macro = 0.5 * (
    #     #     outer(e_l, e_m) +
    #     #     outer(e_m, e_l)
    #     # )

    #     # --------------------------------------------------
    #     # RHS
    #     #
    #     # a(X,v) =
    #     # - integral c:A_macro : grad(v)
    #     # --------------------------------------------------

    #     i, j, r, t = indices(4)

    #     sigma_macro = as_tensor(
    #         sum(
    #             self.cst[i, j, r, t]
    #             * A_macro[r, t]
    #             for r in range(self.dim)
    #             for t in range(self.dim)
    #         ),
    #         (i, j)
    #     )
        
    #     print("||sigma_macro|| = ", norm(sigma_macro))

    #     v = TestFunction(self.P1)

    #     L = -inner(sigma_macro, self.symgrad(v)) * dx
    #     b = assemble(L)

    #     X = Function(self.P1)


    #     self.solver.solve(X, b)

    #     X.rename(f"X_{l}{m}")

    #     # Microscopic strain
    #     eps = 0.5*(A_macro + A_macro.T) + self.symgrad(X)

    #     # Microscopic stress
    #     sigma = as_tensor(
    #         sum(
    #             self.cst[i, j, r, t]
    #             * eps[r, t]
    #             for r in range(self.dim)
    #             for t in range(self.dim)
    #         ),
    #         (i, j)
    #     )

    #     # --------------------------------------------------
    #     # Store stress as DG0
    #     # --------------------------------------------------

    #     sigma_fun = Function(
    #         self.gradFEspace,
    #         name=f"sigma_{l}{m}"
    #     )

    #     sigma_fun.interpolate(sigma)

    #     # Effective tensor component
    #     volume = assemble(
    #         Constant(1.0) * dx(domain=self.mesh)
    #     )

    #     C_lm = np.zeros(
    #         (self.dim, self.dim)
    #     )

    #     for i0 in range(self.dim):
    #         for j0 in range(self.dim):

    #             C_lm[i0, j0] = (assemble(sigma_fun[i0, j0] * dx) / volume)

    #     print("||X|| =",norm(X))
    #     print("||eps|| = ", norm(eps))

    #     # Export
    #     folder = os.path.join(
    #         self.output_dir,
    #         f"cell_{l}{m}"
    #     )

    #     os.makedirs(folder, exist_ok=True)
    #     VTKFile(os.path.join(folder, f"X_{l}{m}.pvd")).write(X)
    #     VTKFile(os.path.join(folder, f"sigma_{l}{m}.pvd")).write(sigma_fun)
    #     np.save(os.path.join(folder, f"C_{l}{m}.npy"), C_lm)

    #     return X, sigma_fun, C_lm


    # def solve_all(self):

    #     Ceff = np.zeros((self.dim, self.dim, self.dim, self.dim))

    #     solutions = {}

    #     for l in range(self.dim):
    #         for m in range(self.dim):
    #             X, sigma, C_lm = self.solve(l, m)
    #             solutions[(l, m)] = (X, sigma)
    #             Ceff[:, :, l, m] = C_lm

    #     self.Ceff = Ceff

    #     np.save(os.path.join(self.output_dir, "Ceff.npy"), Ceff)

    #     return Ceff, solutions


    #     # E_macro = 0.5 * (A_macro + A_macro.T)
    #     # E_macro_norm = sqrt(assemble(inner(E_macro, E_macro) * dx(domain=self.mesh)))
    #     # E_corrector = self.symgrad(X)
    #     # E_total = E_macro + E_corrector
    #     # print("||E_macro||      =", E_macro_norm)
    #     # print("||E_corrector||  =", norm(E_corrector))
    #     # print("||E_total||      =", norm(E_total))

    #     # volume = assemble(
    #     #     Constant(1.0) * dx(domain=self.mesh)
    #     # )

    #     # E_corrector_avg = np.zeros((self.dim, self.dim))

    #     # for iii in range(self.dim):
    #     #     for jjj in range(self.dim):
    #     #         E_corrector_avg[iii, jjj] = (
    #     #             assemble(
    #     #                 E_corrector[iii, jjj] *
    #     #                 dx(domain=self.mesh)
    #     #             )
    #     #             / volume
    #     #         )

    #     # print("<E_corrector> =")
    #     # print(E_corrector_avg)