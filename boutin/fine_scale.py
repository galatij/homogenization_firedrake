from firedrake import *
from ufl import as_tensor, as_matrix
import numpy as np
import os


class FineScaleProblem:

    def __init__(self, mu, lmbda, f_fun, dim, fs_geom):
        nx = fs_geom["nx"]
        ny = fs_geom["ny"]
        nz = fs_geom["nz"]
        n_cell = fs_geom["ncell"]
        L_cell = fs_geom["Lcell"]

        if dim == 3:
            self.mesh = BoxMesh(
                nx*n_cell, ny*n_cell, nz*n_cell, nx*L_cell, ny*L_cell, nz*L_cell,
                hexahedral=True
            )
        elif dim == 2:
            self.mesh = RectangleMesh(
                nx*n_cell, ny*n_cell, nx*L_cell, ny*L_cell,
                quadrilateral=True
            )
        else:
            raise ValueError("dim must be 2 or 3")
        
        self.dim = dim
        self.cst = {}
        self.mu_fun = mu
        self.lmbda_fun = lmbda
        self.f_fun = f_fun

        self._build_spaces()
        self._build_variational_problem()
        self._build_solver()


    def _build_spaces(self):
        self.P0 = FunctionSpace(self.mesh, "DG", 0)
        self.P1 = VectorFunctionSpace(self.mesh, "CG", 1)
        if self.dim == 3:
            self.Rspace = VectorFunctionSpace(self.mesh,"CG",1)
        else:
            self.Rspace = FunctionSpace(self.mesh,"CG",1)
        self.mixed = self.P1 * self.Rspace

    def _build_variational_problem(self):
        # data
        if self.dim == 3:
            x, y, z = SpatialCoordinate(self.mesh)
        else:
            x,y = SpatialCoordinate(self.mesh)
            z = Constant(0.)

        mu_expr = self.mu_fun(x,y,z)
        lmbda_expr = self.lmbda_fun(x,y,z)

        self.mu = Function(self.P0).interpolate(mu_expr)
        self.lmbda = Function(self.P0).interpolate(lmbda_expr)

        self.mu.rename("mu")
        self.lmbda.rename("lmbda")
        VTKFile("output/mu_fs.pvd").write(self.mu)
        VTKFile("output/lmbda_fs.pvd").write(self.lmbda)


        i, j, r, t = indices(4)
        kron = Identity(self.dim)
        self.cst = as_tensor(
            2*self.mu*kron[i,r]*kron[j,t] + self.lmbda*kron[i,j]*kron[r,t],
            (i, j, r, t)
        )

        # variational forms
        u, r = TrialFunctions(self.mixed)
        u_test, r_test = TestFunctions(self.mixed)

        epsilon = grad(u) + self.vskw(r)

        sigma = as_tensor(
            sum(self.cst[i,j,r,t]*epsilon[r,t] for r in range(self.dim) for t in range(self.dim)),
            (i,j)
        )

        self.a = inner(sigma, grad(u_test))*dx + dot(self.mskw(epsilon), r_test)*dx


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

        A = assemble(
            self.a,
            mat_type="aij",
            bcs=bcs,
        )

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

        self.solver.solve(w, b)

        U,R = w.subfunctions

        self.export_solution(U, R)
    
        return U,R

    def export_solution(
        self,
        U, R
    ):
        folder = os.path.join("output/finescale")
        os.makedirs(folder, exist_ok=True)

        U.rename("U")
        R.rename("R")

        VTKFile(os.path.join(folder, f"U.pvd")).write(U)
        VTKFile(os.path.join(folder, f"R.pvd")).write(R)


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
            