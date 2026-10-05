from firedrake import *
from ufl import as_tensor, as_matrix
import numpy as np
import os

class EffectiveProblem:

    def __init__(self, mesh, ell, Ceff, f, dim, order = 2):

        print("\nInitializing effective primal problem...")

        self.ell = ell
        self.mesh = mesh
        self.dim = dim
        self.f_fun = f
        self.order = order

        self.C0 = Ceff[0]

        if order >= 4:
            self.C1 = Ceff[1]
            self.C2 = Ceff[2]
        else:
            self.C1 = None
            self.C2 = None

        self._build_spaces()
        self._build_variational_problem()
        self._build_solver()

        print(" done.")

    def _build_spaces(self):

        # Displacement
        self.Uspace = VectorFunctionSpace(self.mesh, "CG", 3)

        if self.order == 2:
            return

        # Auxiliary strain eps_{ij}
        self.Espace = TensorFunctionSpace(
            self.mesh, "CG", 2,
            shape=(self.dim, self.dim)
        )

        # Gradient of strain  g_{ijk}
        self.Gspace = TensorFunctionSpace(
            self.mesh, "CG", 1,
            shape=(self.dim, self.dim, self.dim)
        )

        # Mixed space (u, eps, g, eta, zeta)
        self.Wspace = MixedFunctionSpace([
            self.Uspace,
            self.Espace,
            self.Gspace
        ])


    # ======================================================
    # Variational problem
    # ======================================================
    def _build_variational_problem(self):

        # Trial functions
        w = TrialFunction(self.Wspace)
        u, Du_r, L_mult = split(w)

        # Test functions
        w_test = TestFunction(self.Wspace)
        u_test, Du_r_test, L_mult_test = split(w_test)

        # Classical stress
        sigma = self._build_classical_stress(u)

        # High-order stress
        H = self._build_high_order_stress(Du_r)

        # -------------------------------------------------
        # 1. Displacement equation
        # -------------------------------------------------

        a_u = inner(
            sigma,
            self.symgrad(u_test)
        ) * dx

        a_u += inner(
            L_mult,
            grad(u_test)
        ) * dx

        # -------------------------------------------------
        # 2. Relaxed-strain equation
        # -------------------------------------------------

        a_r = inner(
            H,
            self.RStrainGrad(Du_r_test)
        ) * dx

        a_r += inner(
            L_mult,
            Du_r_test
        ) * dx

        # -------------------------------------------------
        # 3. Lagrange-multiplier constraint
        # -------------------------------------------------

        a_lambda = inner(
            Du_r - grad(u),
            L_mult_test
        ) * dx

        # -------------------------------------------------
        # Complete bilinear form
        # -------------------------------------------------

        self.a = a_u + a_r + a_lambda

    # def _build_variational_problem(self):

    #     if self.order == 2:
    #         u = TrialFunction(self.Uspace)
    #         u_test = TestFunction(self.Uspace)

    #         Sigma = self._build_stress2(u)
    #     elif self.order == 4:
    #         w = TrialFunction(self.Wspace)
    #         u, eps, g = split(w)
            
    #         w_test = TestFunction(self.Wspace)
    #         u_test, eps_test, g_test = split(w_test)

    #         Sigma = self._build_stress4(eps, g)

    #     self.Sigma = Sigma

    #     # Equilibrium
    #     # int Sigma : grad(v)
    #     self.a = inner(
    #         Sigma,
    #         grad(u_test)
    #     ) * dx

    #     if self.order >= 3:
    #         # Constraint:
    #         # eps = self.symgrad(u)
    #         a_constraint_eps = inner(
    #             eps - self.symgrad(u),
    #             eps_test
    #         ) * dx
    #         self.a += a_constraint_eps

    #     if self.order >= 4:
    #         # Constraint:
    #         # g = grad(eps)
    #         a_constraint_g = inner(
    #             g - grad(eps),
    #             g_test
    #         ) * dx
    #         self.a += a_constraint_g

    # ======================================================
    # Solver
    # ======================================================

    def _build_solver(self):

        if self.order == 2:
            self.bc_u = DirichletBC(
                self.Uspace,
                Constant(np.zeros(self.dim)),
                1
            )

        elif self.order == 4:
            self.bc_u = DirichletBC(
                self.Wspace.sub(0),
                Constant(np.zeros(self.dim)),
                1
            )

        bcs = [self.bc_u]
        
        print("Assembling...")
        
        A = assemble(
            self.a,
            mat_type="aij",
            bcs=bcs
        )

        print(" done.")

        self.solver = LinearSolver(
            A,
            nullspace=None,
            solver_parameters={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps"
            }
        )            

    # ======================================================
    # Solve
    # ======================================================

    def solve(self):

        # Coordinates
        if self.dim == 3:
            x, y, z = SpatialCoordinate(self.mesh)
        else:
            x, y = SpatialCoordinate(self.mesh)
            z = Constant(0.)

        # Body force
        f_expr = self.f_fun(x, y, z)
        force = Function(self.Uspace).interpolate(f_expr)

        if self.order == 2:
            W_test = TestFunction(self.Uspace)
            L = dot(force, W_test) * dx
            W = Function(self.Uspace)

        elif self.order == 4:
            W_test = TestFunction(self.Wspace)
            U_test, _, _ = split(W_test)
            L = dot(force, U_test) * dx
            W = Function(self.Wspace)

        b = assemble(L)

        print("Solving effective primal problem...")

        self.solver.solve(W, b)

        print(" done.")

        if self.order == 2:
            sigma_expr = self._build_stress2(W)

        elif self.order == 4:
            U, Eps, G = W.subfunctions
            sigma_expr = self._build_stress4(Eps, G)

        Sigma = Function(
            TensorFunctionSpace(
                self.mesh,
                "DG",
                0
            ),
            name="Sigma_eff"
        )
    
        Sigma.interpolate(
            sigma_expr
        )

        if self.order == 2:
            # Export
            self.export_solution(W, Sigma)
            
            return W, Sigma

        elif self.order == 4:
            self.export_solution(U, Eps, G, Sigma)

            return U, Eps, G, Sigma


    def symgrad(self, u):

        return 0.5 * (
            grad(u) + grad(u).T
        )

    def RStrain(self, u):
        return grad(u) # or grad(U).T

    def RStrainGrad(self, R):
        i, j, k = indices(3)

        return as_tensor(
            0.5 * (
                R[j, k].dx(i)
                +
                R[i, k].dx(j)
            ),
            (i, j, k)
        )

    def _build_classical_stress(self, u):
        eps = self.symgrad(u)
        Sigma = as_tensor(
            [
                [
                    sum(
                        self.C0[(r, t)][i, j] * eps[r, t]
                        for r in range(self.dim)
                        for t in range(self.dim)
                    )
                    for j in range(self.dim)
                ]
                for i in range(self.dim)
            ]
        )
        return Sigma

    def _build_high_order_stress(self, R):

        RG = self.RStrainGrad(R)
        i, j, k = indices(3)
        HStress = as_tensor(
            [
                [
                    sum(
                        self.C2[(r, t, k, s)][i, j] * (RG[r, t, k] - RG[k, t, r])
                        for r in range(self.dim)
                        for t in range(self.dim)
                        for k in range(self.dim)
                    )
                    for j in range(self.dim)
                ]
                for i in range(self.dim)
            ]
        )
        return HStress

        return as_tensor(
            self.mu * self.ell**2 * (RG[i, j, k] - RG[k, j, i]),
            (i, j, k)
        )
    
    def _build_stress2(self, U):
        return self._build_classical_stress(U)

    def _build_stress4(self, eps, g):

        # C0 : eps
        Sigma0 = as_tensor(
            [
                [
                    sum(
                        self.C0[(r, t)][i, j] * eps[r, t]
                        for r in range(self.dim)
                        for t in range(self.dim)
                    )
                    for j in range(self.dim)
                ]
                for i in range(self.dim)
            ]
        )

        if self.order < 4:
            return Sigma0

        # C1 : g
        Sigma1 = as_tensor(
            [
                [
                    sum(
                        self.C1[(r, t, k)][i, j] * g[r, t, k]
                        for r in range(self.dim)
                        for t in range(self.dim)
                        for k in range(self.dim)
                    )
                    for j in range(self.dim)
                ]
                for i in range(self.dim)
            ]
        )

        # C2 : grad(g)
        grad_g = grad(g)
        Sigma2 = as_tensor(
            [
                [
                    sum(
                        self.C2[(r, t, k, l)][i, j] * grad_g[r, t, k, l]
                        for r in range(self.dim)
                        for t in range(self.dim)
                        for k in range(self.dim)
                        for l in range(self.dim)
                    )
                    for j in range(self.dim)
                ]
                for i in range(self.dim)
            ]
        )

        return Sigma0 + Sigma1 + Sigma2

    def export_solution(
        self,
        U,
        Sigma,
        Eps=None,
        G=None
    ):

        folder = os.path.join("output/effective_primal")

        os.makedirs(folder, exist_ok=True)

        U.rename("U_eff_primal")
        Sigma.rename("Sigma_eff_primal")
        VTKFile(os.path.join(folder, "U_eff.pvd")).write(U)
        VTKFile(os.path.join(folder, "Sigma_eff.pvd")).write(Sigma)

        if Eps is not None:
            Eps.rename("Eps_eff")
            VTKFile(os.path.join(folder, "Eps_eff.pvd")).write(Eps)

        if G is not None:
            G.rename("G_eff")
            VTKFile(os.path.join(folder, "G_eff.pvd")).write(G)
