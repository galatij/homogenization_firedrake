from firedrake import *
from ufl import as_tensor, as_matrix
import numpy as np
import os


class EffectiveMixed:

    def __init__(self, mesh, Ceff, tCeff, f, dim):
        print("\nInitializing effective problem o lowest order...")
        self.mesh = mesh
        self.dim = dim
        self.Ceff = Ceff
        self.tCeff = tCeff
        self.f_fun = f

        self._build_spaces()
        self._build_variational_problem()
        self._build_solver()
        print(" done.")


    def _build_spaces(self):
        self.P1 = VectorFunctionSpace(self.mesh, "CG", 1)
        if self.dim == 3:
            self.Rspace = VectorFunctionSpace(self.mesh,"CG",1)
        else:
            self.Rspace = FunctionSpace(self.mesh,"CG",1)
        self.mixed = self.P1 * self.Rspace

    def _build_variational_problem(self):

        U, R = TrialFunctions(self.mixed)
        V, Q = TestFunctions(self.mixed)

        eps = grad(U) + self.vskw(R)

        # effective stress
        # Ceff2 = as_tensor(self._dict_to_tensor4(self.Ceff[2]))
        # tCeff2 = as_tensor(self._dict_to_tensor4(self.tCeff[2]))

        Ceff2 = as_tensor(self.Ceff)
        tCeff2 = as_tensor(self.tCeff)

        i,j,r,t = indices(4)

        Sigma = as_tensor(
            Ceff2[i,j,r,t]*grad(U)[r,t] + tCeff2[i,j,r,t]*self.vskw(R)[r,t],
            (i,j)
        )


        self.U = U
        self.R = R

        self.V = V
        self.Q = Q

        self.a = (inner(Sigma,grad(V)) + dot(self.mskw(eps),Q)) * dx


    def _build_solver(self):
        self.bc_u = DirichletBC(
            self.mixed.sub(0),
            Constant(np.zeros(self.dim)),
            1
        )

        if self.dim == 3:
            self.bc_r = DirichletBC(
                self.mixed.sub(1),
                Constant(np.zeros(self.dim)),
                1
            )
        else:
            self.bc_r = DirichletBC(
                self.mixed.sub(1),
                Constant(0.),
                1
            )

        bcs=[self.bc_u, self.bc_r]

        print("Assembling...")
        A = assemble(
            self.a,
            mat_type="aij",
            bcs=bcs,
        )
        print(" done.")

        self.solver = LinearSolver(
            A,
            nullspace=None,
            solver_parameters={
                "ksp_type":"preonly",
                "pc_type":"lu",
                "pc_factor_mat_solver_type":"mumps"
            }
        )



    def solve(self):

        V,Q = TestFunctions(self.mixed)
        if self.dim == 3:
            x, y, z = SpatialCoordinate(self.mesh)
        else:
            x, y = SpatialCoordinate(self.mesh)
            z = Constant(0.)

        f_expr = self.f_fun(x,y,z)
        force = Function(self.P1).interpolate(f_expr)

        L = dot(force,V)*dx
        b = assemble(L)

        w = Function(self.mixed)

        print("Solving effective problem...")
        self.solver.solve(w, b)
        print(" done.")

        U,R = w.subfunctions

        self.export_solution(U, R, self.Ceff, self.tCeff)
    
        return U,R

    def export_solution(
        self,
        U, R,
        Ceff, tCeff
    ):
        folder = os.path.join("output/effective_mixed")
        os.makedirs(folder, exist_ok=True)

        U.rename("U_eff_mixed")
        R.rename("R_eff_mixed")

        VTKFile(os.path.join(folder, f"U_eff.pvd")).write(U)
        VTKFile(os.path.join(folder, f"R_eff.pvd")).write(R)


    def _dict_to_tensor4(self, Cdict):
        C = np.zeros((self.dim,self.dim,self.dim,self.dim))

        for (l, m), C_lm in Cdict.items():
            C[:, :, l, m] = np.asarray(C_lm)

        return C

    def mskw(self, A):
        if self.dim == 3:
            return as_vector([
                A[2,1] - A[1,2],
                A[0,2] - A[2,0],
                A[1,0] - A[0,1]
            ])
        else:
            return A[1,0] - A[0,1]

    def vskw(self, v):
        if self.dim ==3:
            return as_matrix([
                [    0, -v[2],  v[1]],
                [ v[2],     0, -v[0]],
                [-v[1],  v[0],     0]
            ])
        else:
            return as_matrix([
                [0,-v],
                [v,0]
            ])




class EffectivePrimal:

    def __init__(self, mesh, Ceff, f, dim):

        print("\nInitializing effective primal problem...")

        self.mesh = mesh
        self.dim = dim
        self.Ceff = Ceff
        self.f_fun = f

        self._build_spaces()
        self._build_variational_problem()
        self._build_solver()

        print(" done.")

    # ======================================================
    # Function spaces
    # ======================================================

    def _build_spaces(self):

        self.P1 = VectorFunctionSpace(
            self.mesh,
            "CG",
            1
        )

    # ======================================================
    # Variational problem
    # ======================================================

    def _build_variational_problem(self):

        U = TrialFunction(self.P1)
        V = TestFunction(self.P1)

        # --------------------------------------------------
        # Effective elasticity tensor
        # --------------------------------------------------

        Ceff = as_tensor(
            self.Ceff
        )

        i, j, r, t = indices(4)

        # --------------------------------------------------
        # Effective stress
        # --------------------------------------------------

        Sigma = as_tensor(
            Ceff[i, j, r, t]
            * self.symgrad(U)[r, t],
            (i, j)
        )

        self.U = U
        self.V = V

        self.Sigma = Sigma

        # --------------------------------------------------
        # Bilinear form
        # --------------------------------------------------

        self.a = (
            inner(
                Sigma,
                self.symgrad(V)
            )
            * dx
        )

    # ======================================================
    # Solver
    # ======================================================

    def _build_solver(self):

        self.bc_u = DirichletBC(
            self.P1,
            Constant(
                np.zeros(self.dim)
            ),
            1
        )

        bcs = [
            self.bc_u
        ]

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

        V = TestFunction(
            self.P1
        )

        if self.dim == 3:

            x, y, z = SpatialCoordinate(
                self.mesh
            )

        else:

            x, y = SpatialCoordinate(
                self.mesh
            )

            z = Constant(0.)

        # --------------------------------------------------
        # Body force
        # --------------------------------------------------

        f_expr = self.f_fun(
            x,
            y,
            z
        )

        force = Function(
            self.P1
        ).interpolate(
            f_expr
        )

        L = dot(
            force,
            V
        ) * dx

        b = assemble(
            L
        )

        # --------------------------------------------------
        # Solve
        # --------------------------------------------------

        U = Function(
            self.P1
        )

        print(
            "Solving effective primal problem..."
        )

        self.solver.solve(
            U,
            b
        )

        print(" done.")

        # --------------------------------------------------
        # Effective stress
        # --------------------------------------------------

        sigma_expr = self.stress(
            U
        )

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

        # --------------------------------------------------
        # Export
        # --------------------------------------------------

        self.export_solution(
            U,
            Sigma,
            self.Ceff
        )

        return U, Sigma

    # ======================================================
    # Symmetric gradient
    # ======================================================

    def symgrad(self, U):

        return 0.5 * (
            grad(U) + grad(U).T
        )

    # ======================================================
    # Stress
    # ======================================================

    def stress(self, U):

        eps = self.symgrad(
            U
        )

        Ceff = as_tensor(
            self.Ceff
        )

        i, j, r, t = indices(4)

        return as_tensor(
            Ceff[i, j, r, t]
            * eps[r, t],
            (i, j)
        )

    # ======================================================
    # Export
    # ======================================================

    def export_solution(
        self,
        U,
        Sigma,
        Ceff
    ):

        folder = os.path.join(
            "output/effective_primal"
        )

        os.makedirs(
            folder,
            exist_ok=True
        )

        # --------------------------------------------------
        # Rename
        # --------------------------------------------------

        U.rename(
            "U_eff_primal"
        )

        Sigma.rename(
            "Sigma_eff_primal"
        )

        # --------------------------------------------------
        # VTK
        # --------------------------------------------------

        VTKFile(
            os.path.join(
                folder,
                "U_eff.pvd"
            )
        ).write(
            U
        )

        VTKFile(
            os.path.join(
                folder,
                "Sigma_eff.pvd"
            )
        ).write(
            Sigma
        )

        # --------------------------------------------------
        # Effective tensor
        # --------------------------------------------------

        np.save(
            os.path.join(
                folder,
                "Ceff.npy"
            ),
            Ceff
        )
