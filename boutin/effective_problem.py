from firedrake import *
from ufl import as_tensor, as_matrix
import numpy as np
import os


class EffectiveElasticityProblem:

    def __init__(self, mesh, Ceff, tCeff, f):
        self.mesh = mesh
        self.Ceff = Ceff
        self.tCeff = tCeff
        self.f_fun = f

        self._build_spaces()
        self._build_variational_problem()
        self._build_solver()


    def _build_spaces(self):
        self.P1 = VectorFunctionSpace(self.mesh, "CG", 1)
        self.mixed = self.P1 * self.P1

    def _build_variational_problem(self):

        U, R = TrialFunctions(self.mixed)
        V, Q = TestFunctions(self.mixed)

        def vskew(v):
            return as_matrix([
                [0,    -v[2],  v[1]],
                [v[2], 0,    -v[0]],
                [-v[1],v[0], 0]
            ])

        def mskew(A):
            return as_vector([
                A[2,1]-A[1,2],
                A[0,2]-A[2,0],
                A[1,0]-A[0,1]
            ])

        eps = grad(U) + vskew(R)

        # effective stress
        # i,j,r,t = indices(4)

        # Sigma = as_tensor(
        #     self.Ceff[i,j,r,t]*grad(U)[r,t] + self.tCeff[i,j,r,t]*vskew(R)[r,t],
        #     (i,j)
        # )

        Ceff2 = self.Ceff[2]
        tCeff2 = self.tCeff[2]

        Sigma = as_tensor([
            [
                sum(
                    as_tensor(Ceff2[(l,m)])[i,j] * grad(U)[l,m]
                    + as_tensor(tCeff2[(l,m)])[i,j] * vskew(R)[l,m]
                    for l in range(3)
                    for m in range(3)
                )
                for j in range(3)
            ]
            for i in range(3)
        ])

        self.U = U
        self.R = R

        self.V = V
        self.Q = Q

        self.a = (inner(Sigma,grad(V)) + dot(mskew(eps),Q)) * dx


    def _build_solver(self):
        self.bc_u = DirichletBC(
            self.mixed.sub(0),
            Constant((0.0,0.0,0.0)),
            1
        )

        self.bc_r = DirichletBC(
            self.mixed.sub(1),
            Constant((0,0,0)),
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
        
        x, y, z = SpatialCoordinate(self.mesh)
        f_expr = self.f_fun(x,y,z)
        force = Function(self.P1).interpolate(f_expr)

        L = dot(force,V)*dx
        b = assemble(L)

        w = Function(self.mixed)

        self.solver.solve(w, b)

        U,R = w.subfunctions

        self.export_solution(U, R, self.Ceff, self.tCeff)
    
        return U,R

    def export_solution(
        self,
        U, R,
        Ceff, tCeff
    ):
        folder = os.path.join("effective")
        os.makedirs(folder, exist_ok=True)

        U.rename("U")
        R.rename("R")

        VTKFile(os.path.join(folder, f"U.pvd")).write(U)
        VTKFile(os.path.join(folder, f"R.pvd")).write(R)
