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
                self.mesh = PeriodicRectangleMesh(n_micro_eff*nx_ref, n_micro_eff*ny_ref, Lx, Ly)
        else:
            if self.dim == 3:
                self.mesh = BoxMesh(n_micro_eff*nx_ref, n_micro_eff*ny_ref, n_micro_eff*nz_ref, Lx, Ly, Lz)
            else:
                self.mesh = RectangleMesh(n_micro_eff*nx_ref, n_micro_eff*ny_ref, Lx, Ly)

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


    # ======================================================
    # Variational problem
    # ======================================================
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

        F30 = (
            u_RStrain[0,0] - u[0].dx(0)
        ) * w_Lagrange[0,0] * dx

        F31 = (
            u_RStrain[0,1] - u[1].dx(0)
        ) * w_Lagrange[0,1] * dx

        F32 = (
            u_RStrain[1,0] - u[0].dx(1)
        ) * w_Lagrange[1,0] * dx

        F33 = (
            u_RStrain[1,1] - u[1].dx(1)
        ) * w_Lagrange[1,1] * dx

        F = F1 + F2 + F30 + F31 + F32 + F33

        # lhs_Lag = ((u_RStrain[0,0] - grad(u)[0,0]) * w_Lagrange[0,0]
        #         + (u_RStrain[0,1] - grad(u)[1,0]) * w_Lagrange[0,1]
        #         + (u_RStrain[1,0] - grad(u)[0,1]) * w_Lagrange[1,0]
        #         + (u_RStrain[1,1] - grad(u)[1,1]) * w_Lagrange[1,1]
        #         ) * dx

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
        self.rhs = dot(Constant((0.,0.)), w)*dx + rhs(F)

        # self.a = F
        # self.rhs = dot(force, w) * dx + dot(self.bcs_Neu, w)*dx
        
    
    # ======================================================
    # Solver
    # ======================================================

    def _build_solver(self):
        
        print("Assembling...")
        
        A = assemble(
            self.a,
            mat_type="aij",
            bcs=self.bcs_Dir
        )
        A_petsc = A.M.handle
        P = A.petscmat
        diag = P.getDiagonal()
        d_arr = diag.getArray(readonly=True)

        print("diag min =", d_arr.min())
        print("diag max =", d_arr.max())
        print("diag abs min =", abs(d_arr).min())
        print("diag abs max =", abs(d_arr).max())

        Ap = A.petscmat

        indptr, indices, values = Ap.getValuesCSR()

        import numpy as np

        dense = np.zeros((Ap.getSize()[0], Ap.getSize()[1]))

        for i in range(Ap.getSize()[0]):
            start = indptr[i]
            end = indptr[i+1]

            cols = indices[start:end]
            vals = values[start:end]

            dense[i, cols] = vals
        
        eig = np.linalg.eigvalsh(dense)

        np.set_printoptions(precision=12, suppress=False)

        print("eigenvalues:")
        print(eig)

        print("smallest |eig| =", np.min(np.abs(eig)))
        print("largest |eig| =", np.max(np.abs(eig)))
        print("rank =", np.linalg.matrix_rank(dense))


        AT = A_petsc.transpose()

        D = A_petsc.copy()
        D.axpy(-1.0, AT)

        print(
            "relative asymmetry =",
            D.norm() / A_petsc.norm()
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

    

    # ======================================================
    # Solve
    # ======================================================

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