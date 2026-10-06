from firedrake import *
from ufl import as_tensor, as_matrix
import numpy as np
import os

# # TODO:
# - check and implement nullspace for the periodic BCs
# - check and implement Dirichlet and Neumann boundary conditions for non-periodic BCs
#   (what about the periodic case?)


class PhunPeng:
    def __init__(self, data, f):

        flags = data["flags"]

        print("\nInitializing effective primal problem...")
        fs_geom = data["geometry"]["fs_geom"]
        nx_ref = data["geometry"]["nref_cell"]
        ny_ref = data["geometry"]["nref_cell"]
        nz_ref = data["geometry"]["nref_cell"]
        Lx = fs_geom["Lx_macro"]
        Ly = fs_geom["Ly_macro"]
        Lz = fs_geom["Lz_macro"]
        n_micro_eff = fs_geom["n_micro_eff"]

        self.eps = fs_geom["eps_ratio"]
        self.dim = data["dim"]
        self.f_fun = f
        self.order = data["order"]
        self.is_periodic = flags["is_per"]
        self.RSgrad_type = flags["RSgrad_type"]

        self.mu = data["coefficients"]["mu_dark"]
        self.lmbda = data["coefficients"]["lmbda_dark"]

        if flags["is_per"]:
            if self.dim == 3:
                self.mesh = PeriodicBoxMesh(n_micro_eff*nx_ref, n_micro_eff*ny_ref, n_micro_eff*nz_ref, Lx,Ly, Lz)
            else:
                self.mesh = PeriodicRectangleMesh(n_micro_eff*nx_ref, n_micro_eff*ny_ref, Lx, Ly, quadrilateral=False)
        else:
            if self.dim == 3:
                self.mesh = BoxMesh(n_micro_eff*nx_ref, n_micro_eff*ny_ref, n_micro_eff*nz_ref, Lx, Ly, Lz)
            else:
                self.mesh = RectangleMesh(n_micro_eff*nx_ref, n_micro_eff*ny_ref, Lx, Ly, quadrilateral=False)

        self.bcs_Dir = None # TODO: check analytically
        self.bcs_Neu = Constant(np.zeros(self.dim)) # TODO: check analytically

        self._build_spaces()
        self._build_variational_problem()
        self._build_solver()

        print(" done.")

    def _build_spaces(self):

        # Displacement
        self.Uspace = VectorFunctionSpace(self.mesh, "CG", 2)   # displacement
        self.RSspace = TensorFunctionSpace(self.mesh, "CG", 1)  # relaxed strain
        self.Lspace = TensorFunctionSpace(self.mesh, "DG", 0)   # Lagrange multiplier
        
        self.mixedFEspace = MixedFunctionSpace([self.Uspace, self.RSspace, self.Lspace])

        if self.is_periodic:
            basis = []
            for ii in range(self.dim):
                X = Function(self.Uspace)
                value = np.zeros(self.dim)
                value[ii] = 1
                X.interpolate(Constant(tuple(value)))
                basis.append(X)
            X_nullspace = VectorSpaceBasis(basis)
            X_nullspace.orthonormalize()
            
            self.nullspace = MixedVectorSpaceBasis(
                self.mixedFEspace,
                [
                    X_nullspace,
                    self.mixedFEspace.sub(1),
                    self.mixedFEspace.sub(2),
                ]
            )
        else:
            x, y = SpatialCoordinate(self.mesh)
            u0 = (-4*x*x + 2*x*y + (11*x)/4 + 9*y*y + 6*y + 1)
            v0 = (8*x*x + 2*y*x + 4*y - (29*y*y)/8 + 2)
            psi11 = (2*y - 8*x + 11.0/4.0)
            psi12 = (16*x + 2*y)
            psi21 = (2*x + 18*y + 6)
            psi22 = (2*x - (29*y)/4 + 4)

            self.bcs_Dir = [
                DirichletBC(self.mixedFEspace.sub(0).sub(0), u0, "on_boundary"),
                DirichletBC(self.mixedFEspace.sub(0).sub(1), v0, "on_boundary"),
                DirichletBC(self.mixedFEspace.sub(1).sub(0), psi11, "on_boundary"),
                DirichletBC(self.mixedFEspace.sub(1).sub(1), psi12, "on_boundary"),
                DirichletBC(self.mixedFEspace.sub(1).sub(2), psi21, "on_boundary"),
                DirichletBC(self.mixedFEspace.sub(1).sub(3), psi22, "on_boundary"),
            ]

    def _build_variational_problem(self):

        E = Constant(1000.)
        nu = Constant(0.25)
        Lambda = (E*nu)/(1.0 - nu*nu)
        G = E/(2.0*(1.0+nu))
        mu = G
        epsilon = Constant(0.1)
        # Trial and test functions
        u, u_RStrain, u_Lagrange = TrialFunctions(self.mixedFEspace)
        w, w_RStrain, w_Lagrange = TestFunctions(self.mixedFEspace)

        delta = Identity(self.dim)
        i, j, k, l, A, B, K, L, M = indices(9)
        def StrainT(u):
            return as_tensor(0.5 * (u[i].dx(j) + u[j].dx(i)), (i, j))

        def StressT(u):
            return as_tensor(
                Lambda * StrainT(u)[k, k]*delta[i, j] + 2.0 * mu * StrainT(u)[i, j],
                (i, j)
            )

        def RStrain(u):
            return as_tensor(
                u[j].dx(i),
                (i, j)
            )

        def RStrainGrad(u_RStrain, form="I"):
            if form == "I":
                return as_tensor(u_RStrain[i, k].dx(j), (i, j, k))

            elif form == "II":
                return as_tensor(
                    0.5 * (u_RStrain[j, k].dx(i) + u_RStrain[i, k].dx(j)), (i, j, k))
            else:
                raise ValueError("form must be either 'I' or 'II'")

        def HStress(u_RStrain):
            RGrad = RStrainGrad(u_RStrain, form=self.RSgrad_type)
            return as_tensor(
                mu * epsilon**2 * (RGrad[i, j, k] - RGrad[k, j, i]),
                (i, j, k)
            )

        # Now define the actual variational formulation
        F1   = (inner(StressT(u),StrainT(w))
                 - inner(u_Lagrange, grad(w).T)
                )*dx # - Rho*bf[j]*w[j]*dx - tr[j]*w[j * ds(1)
        
        RGrad_u = RStrainGrad(u_RStrain,form=self.RSgrad_type)
        RGrad_w = RStrainGrad(w_RStrain,form=self.RSgrad_type)
        H = HStress(u_RStrain)

        F2 = (
            sum(
                H[j, i, k] * RGrad_w[j, i, k]
                for i in range(self.dim)
                for j in range(self.dim)
                for k in range(self.dim)
            )
            + inner(u_Lagrange, w_RStrain)
            ) * dx

        F3 = ((u_RStrain[0,0] - grad(u)[0,0]) * w_Lagrange[0,0]
                + (u_RStrain[0,1] - grad(u)[1,0]) * w_Lagrange[0,1]
                + (u_RStrain[1,0] - grad(u)[0,1]) * w_Lagrange[1,0]
                + (u_RStrain[1,1] - grad(u)[1,1]) * w_Lagrange[1,1]
                ) * dx

        F = F1 + F2 + F3

        # Coordinates
        if self.dim == 3:
            x, y, z = SpatialCoordinate(self.mesh)
        else:
            x, y = SpatialCoordinate(self.mesh)
            z = Constant(0.)

        # Body force
        f_expr = self.f_fun(x, y, z)
        force = Function(self.Uspace).interpolate(f_expr)

        self.a = lhs(F)
        self.rhs = dot(force, w)*dx + rhs(F)

        # self.a = F
        # self.rhs = dot(force, w) * dx + dot(self.bcs_Neu, w)*dx

    def _build_solver(self):
        
        print("Assembling...")
        
        A = assemble(
            self.a,
            mat_type="aij",
            bcs=self.bcs_Dir
        )

        self.b = assemble(self.rhs)

        print(" done.")

        nullspc = self.nullspace if self.is_periodic else None

        # spm = {...}
        
        spm = {
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
                "mat_mumps_icntl_14": 1000,
                # "mat_mumps_icntl_22": 1,
                "ksp_view": None,
                "ksp_converged_reason": None,
                "pc_view": None,
            }

        spm = {
            "ksp_type": "preonly",
            "pc_type": "lu",

            "pc_factor_mat_solver_type": "mumps",

            # "mat_mumps_icntl_24": 1,
            "mat_mumps_icntl_14": 1000,

            # "pc_factor_shift_type": "NONZERO",
            # "pc_factor_shift_amount": 1e-8,

            "ksp_view": None,
            "ksp_converged_reason": None,
            "pc_view": None,
        }
        
        self.solver = LinearSolver(
            A,
            nullspace=nullspc,
            solver_parameters=spm
        )

        # self.diagnostics()

    def solve(self):

        x = Function(self.mixedFEspace)

        print("Solving effective problem...")
        
        self.solver.solve(x, self.b)
        U, U_RStrain, U_Lagrange = x.subfunctions

        self.export(U, U_RStrain, U_Lagrange)

        if not self.is_periodic:
            self.compare_with_exact_solution(U, U_RStrain)
        
        print(" done.")

        return U, U_RStrain, U_Lagrange

    def export(self, U_eff, U_RStrain, U_Lagrange, filename="output/effective_ho.pvd"):

        vtk = VTKFile(filename)

        if self.order >= 4:
            U_eff.rename("U_eff")
            U_RStrain.rename("U_RStrain")
            U_Lagrange.rename("U_Lagrange")
            
            vtk.write(
                U_eff,
                U_RStrain,
                U_Lagrange
            )

        else:
            U_eff.rename("U_eff")

            vtk.write(U_eff)

        print(f"Effective solution exported to {filename}")

    def strain(self, u):
        return 0.5*(grad(u) + grad(u).T)

    def stress(self, u):
        strain = self.strain(u)
        Sigma = as_tensor(
            [
                [
                    sum(
                        self.C0[(r, t)][i, j] * strain[r, t]
                        for r in range(self.dim)
                        for t in range(self.dim)
                    )
                    for j in range(self.dim)
                ]
                for i in range(self.dim)
            ]
        )
        return Sigma

    def inner3(self, hyperstress, rel_strain_grad):
        i, j, k = indices(3)
        return hyperstress[i,j,k] * rel_strain_grad[i,j,k]

    def gradRS(self, u_RStrain, RSgrad_type = "I"):
        i,j,k = indices(3)
        if RSgrad_type == "I":
            return as_tensor(
                grad(u_RStrain)[j,k,i],
                [i,j,k]
            )

        elif RSgrad_type == "II":
            return as_tensor(0.5*(
                grad(u_RStrain)[j,k,i] + grad(u_RStrain)[k,j,i]
                ),
                [i,j,k] 
            )
            # return 0.5*(grad(u_RStrain + u_RStrain.T))

        else:
            raise NotImplementedError("Only type I and II implemented for the relaxed strain.")

        # return as_tensor(
        #         (1./2.*(u_RStrain[k,j].dx(i) + u_RStrain[i,k].dx(j))),
        #         [i,j,k]
        #     )

    def hyperstress(self, u_RStrain, w_RStrain):
        gradRStrain_trial = self.gradRS(u_RStrain, RSgrad_type = self.RSgrad_type)
        gradRStrain_test = self.gradRS(w_RStrain, RSgrad_type = "I")

        hyperSigma_term = (as_tensor(
            sum(
                self.eps**2 * self.C2[(p,q,r,s)][k,j]
                            * gradRStrain_trial[r,p,q]
                            * gradRStrain_test[s,j,k]
                for p in range(self.dim)
                for q in range(self.dim)
                for r in range(self.dim)
                for s in range(self.dim)
                for k in range(self.dim)
                for j in range(self.dim)
            )
        ))

        return hyperSigma_term

    def compare_with_exact_solution(self, u_num, psi_num, filename="output/effective_ho/exact.pvd"):
        """
        Compare the numerical mixed solution with the manufactured exact solution.

        Exact displacement:
            u0 = -4*x^2 + 2*x*y + 11*x/4 + 9*y^2 + 6*y + 1
            v0 =  8*x^2 + 2*x*y + 4*y - 29*y^2/8 + 2

        Exact relaxed strain:
            psi_ij = d u_j / d x_i

        The exact displacement is quadratic and the exact relaxed strain
        is linear, hence they are exactly representable in:
            Uspace  = CG2 vector
            RSspace = CG1 tensor

        Parameters
        ----------
        numerical_solution : Function
            Solution in self.mixedFEspace.

        output_prefix : str
            Prefix used for exported files.

        Returns
        -------
        errors : dict
            L2 and H1 errors for displacement and relaxed strain.
        """

        x, y = SpatialCoordinate(self.mesh)

        # ------------------------------------------------------------
        # Exact displacement
        # ------------------------------------------------------------

        u0 = (
            -4.0*x*x
            + 2.0*x*y
            + (11.0/4.0)*x
            + 9.0*y*y
            + 6.0*y
            + 1.0
        )

        v0 = (
            8.0*x*x
            + 2.0*x*y
            + 4.0*y
            - (29.0/8.0)*y*y
            + 2.0
        )

        u_exact_expr = as_vector([u0, v0])

        # Interpolate into the same displacement space as the numerical solution
        u_exact = Function(self.Uspace, name="u_exact")
        u_exact.interpolate(u_exact_expr)

        # ------------------------------------------------------------
        # Exact relaxed strain
        #
        # Your constraint is
        #
        #     u_RS[i,j] = d u_j / d x_i
        #
        # ------------------------------------------------------------

        psi11 = 2.0*y - 8.0*x + 11.0/4.0
        psi12 = 16.0*x + 2.0*y
        psi21 = 2.0*x + 18.0*y + 6.0
        psi22 = 2.0*x - (29.0/4.0)*y + 4.0

        psi_exact_expr = as_tensor(
            [[psi11, psi12],
            [psi21, psi22]]
        )

        # Interpolate into the same relaxed-strain space
        psi_exact = Function(self.RSspace, name="psi_exact")
        psi_exact.interpolate(psi_exact_expr)

        # ------------------------------------------------------------
        # Errors
        # ------------------------------------------------------------

        e_u = Function(self.Uspace, name="u_error")
        e_u.assign(u_num - u_exact)

        e_psi = Function(self.RSspace, name="psi_error")
        e_psi.assign(psi_num - psi_exact)

        # L2 errors
        error_u_L2 = sqrt(assemble(inner(e_u, e_u) * dx))
        error_psi_L2 = sqrt(assemble(inner(e_psi, e_psi) * dx))

        # H1 errors
        error_u_H1 = sqrt(
            assemble((
                inner(e_u, e_u)
                + inner(grad(e_u), grad(e_u))
            )* dx)
        )

        error_psi_H1 = sqrt(
            assemble((
                inner(e_psi, e_psi)
                + inner(grad(e_psi), grad(e_psi))
            ) * dx)
        )

        # ------------------------------------------------------------
        # Export exact solutions and errors
        # ------------------------------------------------------------

        vtk = VTKFile(filename)
        u_exact.rename("Exact displacement")
        psi_exact.rename("Exact relaxed strain")

        e_u.rename("Displacement error")
        e_psi.rename("Relaxed strain error")
            
        vtk.write(u_exact, psi_exact, e_u, e_psi)

        # ------------------------------------------------------------
        # Print
        # ------------------------------------------------------------

        print("\n--- Exact solution comparison ---")
        print(f"||u - u_exact||_L2       = {float(error_u_L2):.16e}")
        print(f"||u - u_exact||_H1       = {float(error_u_H1):.16e}")
        print(f"||psi - psi_exact||_L2   = {float(error_psi_L2):.16e}")
        print(f"||psi - psi_exact||_H1   = {float(error_psi_H1):.16e}")

        return {
            "u_L2": float(error_u_L2),
            "u_H1": float(error_u_H1),
            "psi_L2": float(error_psi_L2),
            "psi_H1": float(error_psi_H1),
            "u_exact": u_exact,
            "psi_exact": psi_exact,
            "u_error": e_u,
            "psi_error": e_psi,
        }

    def diagnostics(self):
        # ============================================================
        # DIAGNOSTIC: constraint block B = [B_u | B_RS]
        #
        # Mixed formulation:
        #
        #   - <lambda, grad(u)^T>
        #   + <lambda, u_RStrain>
        #
        # with
        #
        #   u        : CG2 vector
        #   u_RStrain: CG1 tensor
        #   lambda   : DG0 tensor
        #
        # We assemble the two constraint blocks independently,
        # rather than extracting them from the global mixed matrix.
        # ============================================================

        import numpy as np

        print("\n" + "="*70)
        print("CONSTRAINT-BLOCK DIAGNOSTIC")
        print("="*70)


        # ------------------------------------------------------------
        # Helper: PETSc sparse matrix -> dense NumPy array
        # ------------------------------------------------------------

        def petsc_to_dense(A):
            """
            Convert a PETSc sparse matrix to a NumPy dense array.
            Intended only for this small diagnostic problem.
            """
            return A.convert("dense").getDenseArray().copy()


        def matrix_rank_and_svals(A, name, rtol=1e-10):

            s = np.linalg.svd(A, compute_uv=False)

            if len(s) == 0:
                rank = 0
                tol = 0.0
            else:
                tol = rtol * s[0]
                rank = np.count_nonzero(s > tol)

            print(f"\n--- {name} ---")
            print(f"shape      = {A.shape}")
            print(f"rank       = {rank}")
            print(f"nullity    = {min(A.shape) - rank}")
            print(f"tol        = {tol:.6e}")

            print("singular values:")
            print(s)

            return s, rank


        # ============================================================
        # 1. SPACE DIMENSIONS
        # ============================================================

        print("\n--- FE-space dimensions ---")

        print("Uspace   :", self.Uspace.dim())
        print("RSspace  :", self.RSspace.dim())
        print("Lspace   :", self.Lspace.dim())


        # ============================================================
        # 2. FULL u-LAGRANGE CONSTRAINT BLOCK
        #
        # B_u : Lspace x Uspace
        #
        #   B_u(lambda,u) = - <lambda, grad(u)^T>
        #
        # ============================================================

        u_trial = TrialFunction(self.Uspace)
        w_lag = TestFunction(self.Lspace)

        B_u_form = (
            -inner(w_lag, grad(u_trial).T)
        ) * dx

        B_u_mat = assemble(
            B_u_form,
            mat_type="aij"
        )

        P_Bu = B_u_mat.petscmat

        print("\n--- B_u PETSc matrix ---")
        print("type       =", P_Bu.getType())
        print("shape      =", P_Bu.getSize())
        print("nnz        =", P_Bu.getInfo()["nz_used"])

        Bu = petsc_to_dense(P_Bu)

        s_Bu, rank_Bu = matrix_rank_and_svals(
            Bu,
            "B_u",
        )


        # ============================================================
        # 3. FULL u_RStrain-LAGRANGE CONSTRAINT BLOCK
        #
        # B_RS : Lspace x RSspace
        #
        #   B_RS(lambda,u_RS) = <lambda, u_RS>
        #
        # ============================================================

        u_RS_trial = TrialFunction(self.RSspace)
        w_lag = TestFunction(self.Lspace)

        B_RS_form = (
            inner(w_lag, u_RS_trial)
        ) * dx

        B_RS_mat = assemble(
            B_RS_form,
            mat_type="aij"
        )

        P_BRS = B_RS_mat.petscmat

        print("\n--- B_RS PETSc matrix ---")
        print("type       =", P_BRS.getType())
        print("shape      =", P_BRS.getSize())
        print("nnz        =", P_BRS.getInfo()["nz_used"])

        BRS = petsc_to_dense(P_BRS)

        s_BRS, rank_BRS = matrix_rank_and_svals(
            BRS,
            "B_RS",
        )


        # ============================================================
        # 4. COMBINED CONSTRAINT BLOCK
        #
        # B = [ B_u | B_RS ]
        #
        # Both blocks must have the same number of rows, namely
        # dim(Lspace).
        # ============================================================

        print("\n--- Combining constraint blocks ---")

        print("B_u  shape =", Bu.shape)
        print("B_RS shape =", BRS.shape)

        assert Bu.shape[0] == BRS.shape[0], (
            "B_u and B_RS have different numbers of rows!"
        )

        B_total = np.hstack([Bu, BRS])

        print("B_total shape =", B_total.shape)

        s_B, rank_B = matrix_rank_and_svals(
            B_total,
            "B = [B_u | B_RS]",
        )


        # ============================================================
        # 5. REPORT THE MULTIPLIER NULLITY
        # ============================================================

        n_lambda = B_total.shape[0]

        nullity_BT = n_lambda - rank_B

        print("\n--- multiplier nullspace ---")
        print("dim(Lspace)       =", n_lambda)
        print("rank(B)           =", rank_B)
        print("nullity(B^T)      =", nullity_BT)


        # ============================================================
        # 6. LEFT NULLSPACE OF B
        #
        # If rank(B) < dim(Lspace), find the corresponding multiplier
        # modes q satisfying
        #
        #       B^T q = 0.
        #
        # This is particularly useful if nullity(B^T) > 0.
        # ============================================================

        if nullity_BT > 0:

            print("\n--- LEFT NULLSPACE OF B ---")

            U, S, Vh = np.linalg.svd(B_total, full_matrices=True)

            # For B with shape (m,n), columns of U corresponding
            # to zero singular values span ker(B^T).
            Z = U[:, rank_B:]

            print("nullspace basis shape =", Z.shape)

            # Check residual
            residual = np.linalg.norm(B_total.T @ Z)

            print("||B^T Z|| =", residual)

            print("\nNorm of each multiplier null mode:")

            for i in range(Z.shape[1]):
                print(
                    f"mode {i}: "
                    f"||z|| = {np.linalg.norm(Z[:, i]):.6e}"
                )


        # ============================================================
        # 7. SCALAR CG1-DG0 MASS MATRIX
        #
        # This isolates the rank deficiency observed in B_RS.
        #
        # CG1 scalar: 9 DOFs on your 2x2 mesh
        # DG0 scalar: 8 DOFs
        #
        # Expected here from your previous test:
        #
        #       shape = (8,9)
        #       rank  = 7
        #
        # ============================================================

        print("\n" + "="*70)
        print("SCALAR CG1-DG0 MASS-MATRIX DIAGNOSTIC")
        print("="*70)

        V_scalar = FunctionSpace(self.mesh, "CG", 1)
        Q_scalar = FunctionSpace(self.mesh, "DG", 0)

        v_trial = TrialFunction(V_scalar)
        q_test = TestFunction(Q_scalar)

        M_scalar_form = q_test * v_trial * dx

        M_scalar = assemble(
            M_scalar_form,
            mat_type="aij"
        )

        P_M = M_scalar.petscmat

        print("\n--- scalar mass matrix ---")
        print("type       =", P_M.getType())
        print("shape      =", P_M.getSize())
        print("nnz        =", P_M.getInfo()["nz_used"])

        M = petsc_to_dense(P_M)

        s_M, rank_M = matrix_rank_and_svals(
            M,
            "scalar CG1-DG0 mass matrix",
        )


        # ============================================================
        # 8. FINAL SUMMARY
        # ============================================================

        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)

        print(f"dim(Uspace)  = {self.Uspace.dim()}")
        print(f"dim(RSspace) = {self.RSspace.dim()}")
        print(f"dim(Lspace)  = {self.Lspace.dim()}")

        print()
        print(f"rank(B_u)    = {rank_Bu}")
        print(f"rank(B_RS)   = {rank_BRS}")
        print(f"rank(B)      = {rank_B}")
        print(f"nullity(B^T) = {nullity_BT}")

        print()
        print(f"scalar CG1-DG0 rank = {rank_M}")

        print("="*70)

        # ============================================================
        # DIAGNOSTIC: ENERGY BLOCKS
        # ============================================================

        E = Constant(1000.)
        nu = Constant(0.25)
        Lambda = (E*nu)/(1.0 - nu*nu)
        G = E/(2.0*(1.0+nu))
        mu = G
        epsilon = Constant(0.1)

        print("\n" + "="*70)
        print("ENERGY-BLOCK DIAGNOSTIC")
        print("="*70)


        # ------------------------------------------------------------
        # Classical displacement stiffness K_u
        # ------------------------------------------------------------

        u_trial = TrialFunction(self.Uspace)
        w_u = TestFunction(self.Uspace)

        delta = Identity(self.dim)

        E = Constant(1000.)
        nu = Constant(0.25)

        Lambda = (E * nu) / (1.0 - nu * nu)
        mu = E / (2.0 * (1.0 + nu))

        i,j,k = indices(3)

        def StrainT_diag(u):
            return as_tensor(
                0.5 * (u[i].dx(j) + u[j].dx(i)),
                (i, j)
            )


        def StressT_diag(u):
            return as_tensor(
                Lambda * StrainT_diag(u)[k, k] * delta[i, j]
                + 2.0 * mu * StrainT_diag(u)[i, j],
                (i, j)
            )
        
        delta = Identity(self.dim)
        i, j, k, l, A, B, K, L, M = indices(9)
        def StrainT(u):
            return as_tensor(0.5 * (u[i].dx(j) + u[j].dx(i)), (i, j))

        def StressT(u):
            return as_tensor(
                Lambda * StrainT(u)[k, k]*delta[i, j] + 2.0 * mu * StrainT(u)[i, j],
                (i, j)
            )

        def RStrain(u):
            return as_tensor(
                u[j].dx(i),
                (i, j)
            )

        def RStrainGrad(u_RStrain, form="I"):
            if form == "I":
                return as_tensor(u_RStrain[i, k].dx(j), (i, j, k))

            elif form == "II":
                return as_tensor(
                    0.5 * (u_RStrain[j, k].dx(i) + u_RStrain[i, k].dx(j)), (i, j, k))
            else:
                raise ValueError("form must be either 'I' or 'II'")

        def HStress(u_RStrain):
            RGrad = RStrainGrad(u_RStrain, form=self.RSgrad_type)
            return as_tensor(
                mu * epsilon**2 * (RGrad[i, j, k] - RGrad[k, j, i]),
                (i, j, k)
            )


        K_u_form = (
            inner(StressT_diag(u_trial), StrainT_diag(w_u))
        ) * dx

        K_u_mat = assemble(
            K_u_form,
            mat_type="aij"
        )

        Ku = K_u_mat.petscmat.convert("dense").getDenseArray().copy()

        print("\n--- K_u ---")
        print("shape =", Ku.shape)

        eig_Ku = np.linalg.eigvalsh(Ku)

        print("min eigenvalue =", eig_Ku[0])
        print("max eigenvalue =", eig_Ku[-1])

        print("smallest eigenvalues:")
        print(eig_Ku[:10])

        tol_Ku = 1e-10 * max(abs(eig_Ku[-1]), 1.0)

        print(
            "numerical nullity =",
            np.count_nonzero(abs(eig_Ku) <= tol_Ku)
        )


        # ------------------------------------------------------------
        # u_RStrain stiffness K_RS
        # ------------------------------------------------------------

        u_RS_trial = TrialFunction(self.RSspace)
        w_RS = TestFunction(self.RSspace)

        RGrad_u = RStrainGrad(
            u_RS_trial,
            form=self.RSgrad_type
        )

        RGrad_w = RStrainGrad(
            w_RS,
            form=self.RSgrad_type
        )

        H_u = as_tensor(
            mu * epsilon**2 *
            (RGrad_u[i, j, k] - RGrad_u[k, j, i]),
            (i, j, k)
        )

        K_RS_form = (
            sum(
                H_u[j, i, k] * RGrad_w[j, i, k]
                for i in range(self.dim)
                for j in range(self.dim)
                for k in range(self.dim)
            )
        ) * dx

        K_RS_mat = assemble(
            K_RS_form,
            mat_type="aij"
        )

        KRS = K_RS_mat.petscmat.convert("dense").getDenseArray().copy()

        print("\n--- K_RS ---")
        print("shape =", KRS.shape)

        eig_KRS = np.linalg.eigvalsh(KRS)

        print("min eigenvalue =", eig_KRS[0])
        print("max eigenvalue =", eig_KRS[-1])

        print("smallest eigenvalues:")
        print(eig_KRS[:20])

        tol_KRS = 1e-10 * max(abs(eig_KRS[-1]), 1.0)

        print(
            "numerical nullity =",
            np.count_nonzero(abs(eig_KRS) <= tol_KRS)
        )


        print("\n" + "="*70)

        # ============================================================
        # FULL MIXED MATRIX DIAGNOSTIC
        # ============================================================

        print("\n" + "="*70)
        print("FULL MIXED MATRIX DIAGNOSTIC")
        print("="*70)

        A = assemble(
            self.a,
            mat_type="aij",
            bcs=self.bcs_Dir
        )

        P = A.petscmat

        print("\n--- PETSc matrix ---")
        print("type      =", P.getType())
        print("shape     =", P.getSize())
        print("nnz       =", P.getInfo()["nz_used"])

        Ad = P.convert("dense").getDenseArray().copy()

        print("\n--- symmetry ---")

        asym = np.linalg.norm(Ad - Ad.T)
        normA = np.linalg.norm(Ad)

        print("||A - A^T||       =", asym)
        print("relative asymmetry =", asym / normA)

        print("\n--- eigenvalues ---")

        eigA = np.linalg.eigvalsh(Ad)

        print("min eigenvalue =", eigA[0])
        print("max eigenvalue =", eigA[-1])

        print("\nsmallest 30 eigenvalues:")
        print(eigA[:30])

        print("\n--- numerical rank ---")

        for rtol in [1e-8, 1e-10, 1e-12, 1e-14]:

            tol = rtol * max(abs(eigA[-1]), 1.0)

            rank = np.count_nonzero(abs(eigA) > tol)
            nullity = len(eigA) - rank

            print(
                f"rtol={rtol:.0e}: "
                f"rank={rank}, "
                f"nullity={nullity}, "
                f"tol={tol:.3e}"
            )


        # ============================================================
        # EXPLICIT NULLSPACE VECTORS
        # ============================================================

        tol = 1e-10 * max(abs(eigA[-1]), 1.0)

        idx_zero = np.where(abs(eigA) <= tol)[0]

        print("\n--- nullspace ---")
        print("number of numerical zero modes =", len(idx_zero))

        if len(idx_zero) > 0:

            eigvals, eigvecs = np.linalg.eigh(Ad)

            Z = eigvecs[:, idx_zero]

            print("nullspace matrix shape =", Z.shape)

            print(
                "||A Z||_F =",
                np.linalg.norm(Ad @ Z)
            )

            # --------------------------------------------------------
            # Inspect how each null mode is distributed among the
            # three fields.
            #
            # IMPORTANT:
            # The global mixed ordering is not assumed here.
            # We therefore need the actual Firedrake field index sets
            # if we want to split these vectors reliably.
            # --------------------------------------------------------

        print("="*70)

        # ============================================================
        # DOES THE CONSTRAINT CONTROL ker(K_RS)?
        # ============================================================

        print("\n" + "="*70)
        print("K_RS KERNEL VS CONSTRAINT")
        print("="*70)

        # Eigen-decomposition of symmetric K_RS
        eig_KRS, vec_KRS = np.linalg.eigh(KRS)

        tol = 1e-10 * max(abs(eig_KRS[-1]), 1.0)

        idx_kernel = np.where(abs(eig_KRS) <= tol)[0]

        N_RS = vec_KRS[:, idx_kernel]

        print("dim ker(K_RS) =", N_RS.shape[1])

        # Apply B_RS to the K_RS nullspace
        BN = BRS @ N_RS

        print("B_RS @ N_RS shape =", BN.shape)

        s_BN = np.linalg.svd(BN, compute_uv=False)

        print("singular values of B_RS restricted to ker(K_RS):")
        print(s_BN)

        tol_BN = 1e-10 * max(abs(s_BN[0]), 1.0)

        rank_BN = np.count_nonzero(s_BN > tol_BN)

        print("rank(B_RS | ker(K_RS)) =", rank_BN)
        print(
            "uncontrolled K_RS kernel =",
            N_RS.shape[1] - rank_BN
        )

        print("="*70)



    # F30 = (
        #     u_RStrain[0,0] - u[0].dx(0)
        # ) * w_Lagrange[0,0] * dx

        # F31 = (
        #     u_RStrain[0,1] - u[1].dx(0)
        # ) * w_Lagrange[0,1] * dx

        # F32 = (
        #     u_RStrain[1,0] - u[0].dx(1)
        # ) * w_Lagrange[1,0] * dx

        # F33 = (
        #     u_RStrain[1,1] - u[1].dx(1)
        # ) * w_Lagrange[1,1] * dx

        # F = F1 + F2 + F30 + F31 + F32 + F33

    # Classical stress
        # def StrainT(u):
        #     # return as_tensor(
        #     #     (1./2.*(u[i].dx(j) + u[j].dx(i))),
        #     #     [i,j]
        #     # )
        
        # def StressT(u):
        #     # return as_tensor(
        #     #     ...
        #     # )

        # def RStrain(u):
        #     return as_tensor(
        #         (u[j].dx(i)),
        #         [i,j]
        #     )
        
        # def RStrainGrad(u_RStrain):
        #     return as_tensor(
        #         (1./2.*(u_RStrain[j,k].dx(i) + u_RStrain[i,k].dx(j))),
        #         [i,j,k]
        #     )
        
        # def HStress(u_RStrain):
        #     return as_tensor(
        #         mu*self.eps**2 * (
        #             RStrainGrad(u_RStrain)[i,j,k]
        #             - RStrainGrad(u_RStrain)[k,j,i]
        #             ),
        #         [i,j,k]
        #     )
        # F1 = StressT(u)[i,j]*StrainT(w)[i,j]*dx -
        #     u_Lagrange[i,j]*w[j].dx(i)*dx - 
        #     Rho*bf[j]*w[j]*dx - tr[j]*w[j]*ds(1)
        # F2 = HStress(u_RStrain)[j,i,k]*RStrainGrad(w_RStrain)[j,i,k]*dx +
        #     u_Lagrange[i,k]*w_RStrain[i,k]*dx
        # F30 = u_RStrain[0,0]*w_Lagrange[0,0]*dx -
        #     u[0].dx(0)*w_Lagrange[0,0]*dx
        
        # F31 = u_RStrain[0,1]*w_Lagrange[0,1]*dx -
        #     u[1].dx(0)*w_Lagrange[0,1]*dx

        # F32 = u_RStrain[1,0]*w_Lagrange[1,0]*dx -
        #     u[0].dx(1)*w_Lagrange[1,0]*dx

        # F33 = u_RStrain[1,1]*w_Lagrange[1,1]*dx -
        #     u[1].dx(1)*w_Lagrange[1,1]*dx

        # F = F1 + F2 + F30 + F31 + F32 + F33

        # self.a = lhs(F)
        # self.L = rhs(F)






        # print()
        # print("============================================================")
        # print(" DIAGNOSTICS ")
        # print("============================================================")

        # print("dim V_u   =", self.Uspace.dim())
        # print("dim V_uRS   =", self.RSspace.dim())
        # print("dim V_uLag =", self.Lspace.dim())
        # print("total     =", self.mixedFEspace.dim())
        # P = A.petscmat

        # nrows, ncols = P.getSize()
        # nnz = P.getInfo()["nz_used"]

        # print()
        # print(" MATRIX ")
        # print(f"size              = {nrows} x {ncols}")
        # print(f"nonzeros          = {int(nnz)}")
        # print(f"density           = {100.0 * nnz / (nrows*ncols):.3e} %")

        # # ============================================================
        # # Diagonal
        # # ============================================================

        # diag = P.getDiagonal()
        # d = diag.getArray(readonly=True)

        # print()
        # print("--- diagonal ---")
        # print(f"min               = {d.min():.16e}")
        # print(f"max               = {d.max():.16e}")
        # print(f"min abs           = {np.abs(d).min():.16e}")
        # print(f"max abs           = {np.abs(d).max():.16e}")
        # print(f"number exactly 0  = {np.count_nonzero(d == 0.0)}")
        # print(f"number |d| < 1e-14 = "
        #       f"{np.count_nonzero(np.abs(d) < 1e-14)}")

        # # ============================================================
        # # Global matrix norm
        # # ============================================================

        # A_norm = P.norm()

        # print()
        # print("--- matrix norm ---")
        # print(f"||A||_F           = {A_norm:.16e}")

        # # ============================================================
        # # Symmetry
        # # ============================================================

        # AT = P.transpose()

        # D = P.copy()
        # D.axpy(-1.0, AT)

        # asym_abs = D.norm()
        # asym_rel = asym_abs / A_norm if A_norm > 0 else np.nan

        # print()
        # print("--- symmetry ---")
        # print(f"||A-A^T||_F       = {asym_abs:.16e}")
        # print(f"relative asymmetry = {asym_rel:.16e}")

        # # ============================================================
        # # Sparse matrix -> dense matrix
        # #
        # # Only do this for small systems.
        # # ============================================================

        # DENSE_LIMIT = 500

        # if nrows == ncols and nrows <= DENSE_LIMIT:

        #     indptr, indices, values = P.getValuesCSR()

        #     dense = np.zeros((nrows, ncols), dtype=np.float64)

        #     for i in range(nrows):
        #         start = indptr[i]
        #         end = indptr[i + 1]

        #         dense[i, indices[start:end]] = values[start:end]


        #     # ============================================================
        #     # Mixed block structure
        #     # ============================================================

        #     n_u   = self.Uspace.dim()
        #     n_rs  = self.RSspace.dim()
        #     n_lag = self.Lspace.dim()

        #     assert n_u + n_rs + n_lag == nrows

        #     Auu   = dense[:n_u, :n_u]
        #     AuRS  = dense[:n_u, n_u:n_u+n_rs]
        #     AuL   = dense[:n_u, n_u+n_rs:]

        #     ARS_u = dense[n_u:n_u+n_rs, :n_u]
        #     ARSRS = dense[n_u:n_u+n_rs, n_u:n_u+n_rs]
        #     ARSL  = dense[n_u:n_u+n_rs, n_u+n_rs:]

        #     AL_u  = dense[n_u+n_rs:, :n_u]
        #     ALRS  = dense[n_u+n_rs:, n_u:n_u+n_rs]
        #     ALL   = dense[n_u+n_rs:, n_u+n_rs:]


        #     print("\n--- mixed block norms ---")

        #     blocks = {
        #         "Auu": Auu,
        #         "AuRS": AuRS,
        #         "AuL": AuL,
        #         "ARS_u": ARS_u,
        #         "ARSRS": ARSRS,
        #         "ARSL": ARSL,
        #         "AL_u": AL_u,
        #         "ALRS": ALRS,
        #         "ALL": ALL,
        #     }

        #     Bu  = AL_u
        #     BRS = ALRS

        #     su = np.linalg.svd(Bu, compute_uv=False)
        #     sr = np.linalg.svd(BRS, compute_uv=False)

        #     print("rank Bu :", np.sum(su > 1e-10 * su[0]))
        #     print("rank BRS:", np.sum(sr > 1e-10 * sr[0]))

        #     print("singular values Bu:")
        #     print(su)

        #     print("singular values BRS:")
        #     print(sr)

        #     for name, block in blocks.items():
        #         print(f"{name:8s} shape={str(block.shape):12s} "
        #             f"norm={np.linalg.norm(block):.16e}")


        #     # Constraint matrix
        #     B = np.hstack((AL_u, ALRS))

        #     print("\n--- actual constraint block B ---")
        #     print("shape =", B.shape)

        #     sB = np.linalg.svd(B, compute_uv=False)

        #     print("largest singular value =", sB[0])
        #     print("smallest singular value =", sB[-1])

        #     for tol in [1e-8, 1e-10, 1e-12, 1e-14]:
        #         rankB = np.sum(sB > tol * sB[0])
        #         print(
        #             f"rank(B), relative tol={tol:.0e}: "
        #             f"{rankB}/{min(B.shape)}"
        #         )

        #     U, s, Vt = np.linalg.svd(B)
        #     print(s)
        #     tol = 1e-10 * s[0]
        #     rank = np.sum(s > tol)
        #     null_Bt = U[:, rank:]
        #     print("rank(B) =", rank)
        #     print("nullity(B^T) =", null_Bt.shape[1])
        #     # --------------------------------------------------------
        #     # Basic numerical checks
        #     # --------------------------------------------------------

        #     print()
        #     print("--- dense matrix checks ---")
        #     print(f"finite            = {np.isfinite(dense).all()}")
        #     print(f"max |A_ij|        = {np.max(np.abs(dense)):.16e}")

        #     nonzero = np.abs(dense[np.abs(dense) > 0.0])

        #     if len(nonzero) > 0:
        #         print(f"min nonzero |A_ij| = {nonzero.min():.16e}")
        #         print(f"max nonzero |A_ij| = {nonzero.max():.16e}")

        #     # --------------------------------------------------------
        #     # Eigenvalues
        #     # --------------------------------------------------------

        #     eigval, eigvec = np.linalg.eigh(dense)

        #     np.set_printoptions(
        #         precision=12,
        #         suppress=False,
        #         linewidth=200
        #     )

        #     # print()
        #     # print("--- eigenvalues ---")
        #     # print(eigval)

        #     abs_eig = np.abs(eigval)

        #     print()
        #     print("--- spectrum ---")
        #     print(f"min eigenvalue     = {eigval[0]:.16e}")
        #     print(f"max eigenvalue     = {eigval[-1]:.16e}")
        #     print(f"smallest |lambda|  = {abs_eig.min():.16e}")
        #     print(f"largest  |lambda|  = {abs_eig.max():.16e}")

        #     # --------------------------------------------------------
        #     # Count near-zero eigenvalues at several tolerances
        #     # --------------------------------------------------------

        #     print()
        #     print("--- near-zero eigenvalues ---")

        #     for tol in [1e-8, 1e-10, 1e-12, 1e-14]:
        #         nzero = np.count_nonzero(abs_eig < tol)

        #         print(
        #             f"|lambda| < {tol:.0e} : {nzero}"
        #         )

        #     # --------------------------------------------------------
        #     # Numerical rank
        #     # --------------------------------------------------------

        #     print()
        #     print("--- numerical rank ---")

        #     for tol in [1e-8, 1e-10, 1e-12, 1e-14]:
        #         rank = np.linalg.matrix_rank(
        #             dense,
        #             tol=tol
        #         )

        #         print(
        #             f"rank (tol={tol:.0e}) = {rank}"
        #         )

        #     # --------------------------------------------------------
        #     # Condition number from singular values
        #     # --------------------------------------------------------

        #     s = np.linalg.svd(
        #         dense,
        #         compute_uv=False
        #     )

        #     print()
        #     print("--- singular values ---")
        #     print(f"largest singular   = {s[0]:.16e}")
        #     print(f"smallest singular  = {s[-1]:.16e}")

        #     for tol in [1e-8, 1e-10, 1e-12, 1e-14]:
        #         nsmall = np.count_nonzero(s < tol)

        #         print(
        #             f"singular values < {tol:.0e} : {nsmall}"
        #         )

        #     # --------------------------------------------------------
        #     # Nullspace eigenvectors
        #     #
        #     # Since A is symmetric, eigenvectors associated with
        #     # lambda ~= 0 span the numerical nullspace.
        #     # --------------------------------------------------------

        #     null_tol = 1e-10
        #     null_idx = np.where(abs_eig < null_tol)[0]

        #     print()
        #     print("--- numerical nullspace ---")
        #     print(
        #         f"tolerance          = {null_tol:.0e}"
        #     )
        #     print(
        #         f"number of null modes = {len(null_idx)}"
        #     )

        #     for k, j in enumerate(null_idx):

        #         v = eigvec[:, j]

        #         print()
        #         print(
        #             f"null mode {k+1}: "
        #             f"eigenvalue = {eigval[j]:.16e}"
        #         )

        #         print(
        #             f"  ||v||_2       = {np.linalg.norm(v):.16e}"
        #         )

        #         print(
        #             f"  max |v_i|     = {np.max(np.abs(v)):.16e}"
        #         )

        #         # Print the vector itself for the tiny diagnostic case.
        #         print("  vector:")
        #         # print(v)

        # else:

        #     print()
        #     print("--- spectral diagnostics skipped ---")
        #     print(
        #         f"Matrix has {nrows} DOFs; "
        #         f"dense diagnostics disabled for n > {DENSE_LIMIT}."
        #     )

        # print()
        # print("============================================================")
        # print(" END MATRIX DIAGNOSTICS")
        # print("============================================================")
        # print()
