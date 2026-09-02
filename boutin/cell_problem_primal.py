from firedrake import *
from ufl import as_vector, as_matrix
import numpy as np
import os
from firedrake.output import VTKFile


class PrimalCellProblem:

    def __init__(self, n, mu, lmbda, dim, output_dir="output/primal"):

        print("Initializing primal cell problem...")

        self.dim = dim
        self.mu_fun = mu
        self.lmbda_fun = lmbda
        self.output_dir = output_dir

        # Mesh
        if dim == 3:
            self.mesh = PeriodicUnitCubeMesh(n, n, n, hexahedral=True)
        elif dim == 2:
            self.mesh = PeriodicUnitSquareMesh(n, n, quadrilateral=True)
        else:
            raise ValueError("dim must be 2 or 3")

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

        print("Initialization done.")


    def symgrad(self, u):

        return 0.5 * (
            grad(u) + grad(u).T
        )


    def solve(self, l, m):

        """
        Solve

            -div [ c : (e_l x e_m + eps(X)) ] = 0

        for the displacement corrector X^{lm}.
        """

        print(
            f"\nSolving primal cell problem ({l},{m})"
        )

        # --------------------------------------------------
        # Basis tensor A = e_l \otimes e_m
        # --------------------------------------------------

        e_l = np.zeros(self.dim)
        e_m = np.zeros(self.dim)

        e_l[l] = 1.0
        e_m[m] = 1.0

        e_l = as_vector(tuple(e_l))
        e_m = as_vector(tuple(e_m))

        A_macro = outer(e_l, e_m)
        # A_macro = 0.5 * (
        #     outer(e_l, e_m) +
        #     outer(e_m, e_l)
        # )

        # --------------------------------------------------
        # RHS
        #
        # a(X,v) =
        # - integral c:A_macro : grad(v)
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
        
        print("||sigma_macro|| = ", norm(sigma_macro))

        v = TestFunction(self.P1)

        L = -inner(sigma_macro, self.symgrad(v)) * dx
        b = assemble(L)

        X = Function(self.P1)


        self.solver.solve(X, b)

        # E_macro = 0.5 * (A_macro + A_macro.T)
        # E_macro_norm = sqrt(assemble(inner(E_macro, E_macro) * dx(domain=self.mesh)))
        # E_corrector = self.symgrad(X)
        # E_total = E_macro + E_corrector
        # print("||E_macro||      =", E_macro_norm)
        # print("||E_corrector||  =", norm(E_corrector))
        # print("||E_total||      =", norm(E_total))

        # volume = assemble(
        #     Constant(1.0) * dx(domain=self.mesh)
        # )

        # E_corrector_avg = np.zeros((self.dim, self.dim))

        # for iii in range(self.dim):
        #     for jjj in range(self.dim):
        #         E_corrector_avg[iii, jjj] = (
        #             assemble(
        #                 E_corrector[iii, jjj] *
        #                 dx(domain=self.mesh)
        #             )
        #             / volume
        #         )

        # print("<E_corrector> =")
        # print(E_corrector_avg)

        X.rename(f"X_{l}{m}")

        # Microscopic strain
        eps = 0.5*(A_macro + A_macro.T) + self.symgrad(X)

        # Microscopic stress
        sigma = as_tensor(
            sum(
                self.cst[i, j, r, t]
                * eps[r, t]
                for r in range(self.dim)
                for t in range(self.dim)
            ),
            (i, j)
        )

        # --------------------------------------------------
        # Store stress as DG0
        # --------------------------------------------------

        sigma_fun = Function(
            self.gradFEspace,
            name=f"sigma_{l}{m}"
        )

        sigma_fun.interpolate(sigma)

        # Effective tensor component
        volume = assemble(
            Constant(1.0) * dx(domain=self.mesh)
        )

        C_lm = np.zeros(
            (self.dim, self.dim)
        )

        for i0 in range(self.dim):
            for j0 in range(self.dim):

                C_lm[i0, j0] = (assemble(sigma_fun[i0, j0] * dx) / volume)

        print("||X|| =",norm(X))
        print("||eps|| = ", norm(eps))

        # Export
        folder = os.path.join(
            self.output_dir,
            f"cell_{l}{m}"
        )

        os.makedirs(folder, exist_ok=True)
        VTKFile(os.path.join(folder, f"X_{l}{m}.pvd")).write(X)
        VTKFile(os.path.join(folder, f"sigma_{l}{m}.pvd")).write(sigma_fun)
        np.save(os.path.join(folder, f"C_{l}{m}.npy"), C_lm)

        return X, sigma_fun, C_lm


    def solve_all(self):

        Ceff = np.zeros((self.dim, self.dim, self.dim, self.dim))

        solutions = {}

        for l in range(self.dim):
            for m in range(self.dim):
                X, sigma, C_lm = self.solve(l, m)
                solutions[(l, m)] = (X, sigma)
                Ceff[:, :, l, m] = C_lm

        self.Ceff = Ceff

        np.save(os.path.join(self.output_dir, "Ceff.npy"), Ceff)

        return Ceff, solutions