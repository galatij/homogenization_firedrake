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
        # CG2, CG1, CG1 -> fail at ref = 32
        # CG2, DG1, DG1 -> fail at ref = 16
        # CG2, CG1, DG1 / CG2, CG1, DG0 / CG2, DG1, DG0 --> always fail
        
        self.mixedFEspace = MixedFunctionSpace([self.Uspace, self.RSspace, self.Lspace])

        if self.is_periodic:
            # TODO: modify Nullspacebasis = []
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

    # ======================================================
    # Variational problem
    # ======================================================
    def _build_variational_problem(self):

        # Trial and test functions
        u, u_RStrain, u_Lagrange = TrialFunctions(self.mixedFEspace)
        w, w_RStrain, w_Lagrange = TestFunctions(self.mixedFEspace)

        # Now define the actual variational formulation
        lhs_u   = (inner(self.stress(u),self.strain(w))
                 - inner(u_Lagrange, grad(w).T)
                )*dx
        
        # lhs_RS  = (inner(self.hyperstress(u_RStrain), self.gradRS(w_RStrain, self.RSgrad_type))
        #          + inner(u_Lagrange, w_RStrain)
        #         )*dx
        
        lhs_RS  = ( self.hyperstress(u_RStrain, w_RStrain)
                    + inner(u_Lagrange, w_RStrain)
                )*dx

        lhs_Lag = ((u_RStrain[0,0] - grad(u)[0,0]) * w_Lagrange[0,0]
                + (u_RStrain[0,1] - grad(u)[1,0]) * w_Lagrange[0,1]
                + (u_RStrain[1,0] - grad(u)[0,1]) * w_Lagrange[1,0]
                + (u_RStrain[1,1] - grad(u)[1,1]) * w_Lagrange[1,1]
                ) * dx

        # Coordinates
        if self.dim == 3:
            x, y, z = SpatialCoordinate(self.mesh)
        else:
            x, y = SpatialCoordinate(self.mesh)
            z = Constant(0.)

        # BCs: TODO
        if not self.is_periodic:
            pass
        else:
            # if I assume smooth u across the boundary, its derivative are periodic...
            pass

        # Body force
        f_expr = self.f_fun(x, y, z)
        force = Function(self.Uspace).interpolate(f_expr)

        self.a = lhs_u + lhs_RS + lhs_Lag
        self.rhs = dot(force, w) * dx + dot(self.bcs_Neu, w)*dx
        
    
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
                "pc_factor_mat_solver_type": "mumps"
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