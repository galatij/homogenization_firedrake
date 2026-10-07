from firedrake import *
from ufl import as_tensor, as_matrix
import numpy as np
import os

# # TODO:
# - check and implement nullspace for the periodic BCs
# - check and implement Dirichlet and Neumann boundary conditions for non-periodic BCs
#   (what about the periodic case?)


class EffectiveProblemHo:

    def __init__(self, data, Ceff, f):

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
        
        self.C0 = Ceff[0]

        if self.order >= 4:
            self.C1 = Ceff[1]
            self.C2 = Ceff[2]
        else:
            self.C1 = None
            self.C2 = None

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
            u0 = None
            v0 = None
            psi11 = None
            psi12 = None
            psi21 = None
            psi22 = None

            self.bcs_Dir = [
                DirichletBC(self.mixedFEspace.sub(0).sub(0), u0, "on_boundary"),
                DirichletBC(self.mixedFEspace.sub(0).sub(1), v0, "on_boundary"),
                DirichletBC(self.mixedFEspace.sub(1).sub(0), psi11, "on_boundary"),
                DirichletBC(self.mixedFEspace.sub(1).sub(1), psi12, "on_boundary"),
                DirichletBC(self.mixedFEspace.sub(1).sub(2), psi21, "on_boundary"),
                DirichletBC(self.mixedFEspace.sub(1).sub(3), psi22, "on_boundary"),
            ]

    def _build_variational_problem(self):

        # =================  TODO: modify  =================== #
        E = Constant(1000.)
        nu = Constant(0.25)
        Lambda = (E*nu)/(1.0 - nu*nu)
        G = E/(2.0*(1.0+nu))
        mu = G
        epsilon = Constant(0.1)
        # ==================================================== #

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

        # =================  TODO: modify  =================== #
        def HStress(u_RStrain):
            RGrad = RStrainGrad(u_RStrain, form=self.RSgrad_type)
            return as_tensor(
                mu * epsilon**2 * (RGrad[i, j, k] - RGrad[k, j, i]),
                (i, j, k)
            )
        # ==================================================== #
        def hyperstress(u_RStrain, w_RStrain):
            RGrad_u = RStrainGrad(u_RStrain,form=self.RSgrad_type)
            RGrad_w = RStrainGrad(w_RStrain,form=self.RSgrad_type)

            hyperSigma_term = (as_tensor(
                sum(
                    self.eps**2 * self.C2[(p,q,r,s)][k,j]
                                * RGrad_u[r,p,q]
                                * RGrad_w[s,j,k]
                    for p in range(self.dim)
                    for q in range(self.dim)
                    for r in range(self.dim)
                    for s in range(self.dim)
                    for k in range(self.dim)
                    for j in range(self.dim)
                )
            ))

            return hyperSigma_term

        # Now define the actual variational formulation
        F1   = (inner(self.stress(u),StrainT(w))
                 - inner(u_Lagrange, grad(w).T)
                )*dx # - Rho*bf[j]*w[j]*dx - tr[j]*w[j * ds(1)
        
        RGrad_u = RStrainGrad(u_RStrain,form=self.RSgrad_type)
        RGrad_w = RStrainGrad(w_RStrain,form=self.RSgrad_type)
        H = HStress(u_RStrain)

        F2 = (
            # sum(
            #     H[j, i, k] * RGrad_w[j, i, k]
            #     for i in range(self.dim)
            #     for j in range(self.dim)
            #     for k in range(self.dim)
            # )
            hyperstress(u_RStrain, w_RStrain)
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

    def _build_solver(self):
        
        print("Assembling...")
        
        A = assemble(self.a, mat_type="aij", bcs=self.bcs_Dir)
        self.b = assemble(self.rhs)

        print(" done.")

        nullspc = self.nullspace if self.is_periodic else None

        # spm = {...}
        
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
        
        # spm = {
        #     "ksp_type": "gmres",
        #     "ksp_rtol": 1e-8,
        #     "ksp_atol": 1e-12,
        #     "ksp_max_it": 5000,

        #     "pc_type": "ilu",

        #     "ksp_monitor": None,
        #     "ksp_converged_reason": None,
        # }
        self.solver = LinearSolver(
            A,
            nullspace=nullspc,
            solver_parameters=spm
        )
    
    def solve(self):

        x = Function(self.mixedFEspace)

        print("Solving effective problem...")
        
        self.solver.solve(x, self.b)

        U, U_RStrain, U_Lagrange = x.subfunctions

        self.export(U, U_RStrain, U_Lagrange)
        
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


#       
    # def _OLD_build_variational_problem(self):

    #     # Trial and test functions
    #     u, u_RStrain, u_Lagrange = TrialFunctions(self.mixedFEspace)
    #     w, w_RStrain, w_Lagrange = TestFunctions(self.mixedFEspace)

    #     # Now define the actual variational formulation
    #     lhs_u   = (inner(self.stress(u),self.strain(w))
    #              - inner(u_Lagrange, grad(w).T)
    #             )*dx
        
    #     # lhs_RS  = (inner(self.hyperstress(u_RStrain), self.gradRS(w_RStrain, self.RSgrad_type))
    #     #          + inner(u_Lagrange, w_RStrain)
    #     #         )*dx
        
    #     lhs_RS  = ( self.hyperstress(u_RStrain, w_RStrain)
    #                 + inner(u_Lagrange, w_RStrain)
    #             )*dx

    #     lhs_Lag = ((u_RStrain[0,0] - grad(u)[0,0]) * w_Lagrange[0,0]
    #             + (u_RStrain[0,1] - grad(u)[1,0]) * w_Lagrange[0,1]
    #             + (u_RStrain[1,0] - grad(u)[0,1]) * w_Lagrange[1,0]
    #             + (u_RStrain[1,1] - grad(u)[1,1]) * w_Lagrange[1,1]
    #             ) * dx

    #     # Coordinates
    #     if self.dim == 3:
    #         x, y, z = SpatialCoordinate(self.mesh)
    #     else:
    #         x, y = SpatialCoordinate(self.mesh)
    #         z = Constant(0.)

    #     # BCs: TODO
    #     if not self.is_periodic:
    #         pass
    #     else:
    #         # if I assume smooth u across the boundary, its derivative are periodic...
    #         pass

    #     # Body force
    #     f_expr = self.f_fun(x, y, z)
    #     force = Function(self.Uspace).interpolate(f_expr)

    #     self.a = lhs_u + lhs_RS + lhs_Lag
    #     self.rhs = dot(force, w) * dx + dot(self.bcs_Neu, w)*dx
        
    
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