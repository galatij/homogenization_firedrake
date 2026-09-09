from firedrake import *
from ufl import as_tensor, as_matrix
import numpy as np
import os

class EffectiveProblemLinear:

    def __init__(self, mesh, ell, Ceff, f, dim, order = 2, is_per = False):

        print("\nInitializing effective primal problem...")

        self.mesh = mesh
        self.ell = ell
        self.dim = dim
        self.f_fun = f
        self.order = order
        self.is_periodic = is_per

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

        if self.is_periodic:
            # Nullspace
            basis = []
            
            for k in range(self.dim):
                v = Function(self.Uspace)
                value = np.zeros(self.dim)
                value[k] = 1.0
                v.interpolate(Constant(tuple(value)))
                basis.append(v)
            self.nullspace = VectorSpaceBasis(basis)
            self.nullspace.orthonormalize()

    # ======================================================
    # Variational problem
    # ======================================================
    def _build_variational_problem(self):

        # Trial functions
        U = TrialFunction(self.Uspace)
        V = TestFunction(self.Uspace)

        # Classical stress
        Sigma = self._build_stress_C0(U)

        self.a = inner(
            Sigma,
            self.symgrad(V)
        ) * dx


    # ======================================================
    # Solver
    # ======================================================

    def _build_solver(self):

        self.bc_u = DirichletBC(
            self.Uspace,
            Constant(np.zeros(self.dim)),
            1
        )

        bcs = [self.bc_u] if not self.is_periodic else []
        
        print("Assembling...")
        
        A = assemble(
            self.a,
            mat_type="aij",
            bcs=bcs
        )

        print(" done.")

        nullspc = self.nullspace if self.is_periodic else None
        self.solver = LinearSolver(
            A,
            nullspace=nullspc,
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

        U0 = Function(self.Uspace)
        V = TestFunction(self.Uspace)

        L0 = dot(force, V) * dx
        b0 = assemble(L0)

        print("Solving effective problem for U0...")
        
        self.solver.solve(U0, b0)
        
        print(" done.")


        if self.order >= 3:
            U1 = Function(self.Uspace)

            C1_eps0 = self._build_stress_C1(U0)
            
            L1 = -inner(C1_eps0, grad(V)) * dx
            
            bc = [self.bc_u] if not self.is_periodic else []

            b1 = assemble(L1, bcs = bc)

            print("Solving effective problem for U1...")
            
            self.solver.solve(U1, b1)
            
            print(" done.")

        if self.order >= 4:
            U2 = Function(self.Uspace)

            C1_eps1 = self._build_stress_C1(U1)
            C2_eps0 = self._build_stress_C2(U0)
    
            L2 = - inner(C1_eps1 + C2_eps0, grad(V)) * dx

            bc = [self.bc_u] if not self.is_periodic else []
            b2 = assemble(L2, bcs=bc)

            print("Solving effective problem for U2...")
    
            self.solver.solve(U2, b2)
    
            print(" done.")

        U_eff = Function(self.Uspace, name="U_eff")

        if self.order < 4:
            U_eff.assign(U0)

        else:
            U_eff.interpolate(U0 + self.ell * U1 + self.ell**2 * U2)

        if self.order == 2:
            self.export(U_eff, U0)
        elif self.order == 4:
            self.export(U_eff, U0, U1, U2)

        return U_eff

    def export(self, U_eff, U0, U1 = None, U2 = None, filename="output/effective_linear.pvd"):

        # if not hasattr(self, "U_eff"):
        #     raise RuntimeError(
        #         "You must call solve() before export()."
        #     )

        vtk = VTKFile(filename)

        if self.order >= 4:
            U_eff.rename("U_eff")
            U0.rename("U0")
            U1.rename("U1")
            U2.rename("U2")
            
            vtk.write(
                U0,
                U1,
                U2,
                U_eff
            )

        else:
            U_eff.rename("U_eff")
            U0.rename("U0")

            vtk.write(
                U0,
                U_eff
            )

        print(f"Effective solution exported to {filename}")

    def symgrad(self, u):

        return 0.5 * (
            grad(u) + grad(u).T
        )

    def _build_stress_C0(self, u):
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

    def _build_stress_C1(self, u):
        grad_eps = grad(self.symgrad(u))
        Sigma = as_tensor(
            [
                [
                    sum(
                        self.C1[(r, t, k)][i, j] * grad_eps[r, t, k]
                        for r in range(self.dim)
                        for t in range(self.dim)
                        for k in range(self.dim)
                    )
                    for j in range(self.dim)
                ]
                for i in range(self.dim)
            ]
        )
        return Sigma


    def _build_stress_C2(self, u):

        # C2 : grad(g)
        grad_eps = grad(self.symgrad(u))
        gradgrad_eps = grad(grad_eps)
        Sigma = as_tensor(
            [
                [
                    sum(
                        self.C2[(r, t, k, l)][i, j] * gradgrad_eps[r, t, k, l]
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

        return Sigma

    # def export_solution(
    #     self,
    #     U,
    #     Sigma
    # ):

    #     folder = os.path.join("output/effective_primal")

    #     os.makedirs(folder, exist_ok=True)

    #     U.rename("U_eff_primal")
    #     Sigma.rename("Sigma_eff_primal")
    #     VTKFile(os.path.join(folder, "U_eff.pvd")).write(U)
    #     VTKFile(os.path.join(folder, "Sigma_eff.pvd")).write(Sigma)
