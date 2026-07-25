from firedrake import *
from ufl import as_vector, as_matrix
import numpy as np
import os
from firedrake.output import VTKFile
current_path = os.getcwd()
print(current_path)

class CellProblem:
    def __init__(self, n, mu, lmbda):
        self.mesh = PeriodicUnitCubeMesh(n, n, n, hexahedral=True)
        self.cst = {}
        self.CST = {}
        self.CST_np = np.zeros((3,3,3,3))
        self.mu_fun = mu
        self.lmbda_fun = lmbda
        self.beta = 1

        self._build_variational_problem()
        self._build_solver()
        # self._build_solver() # TODO
        
    def _build_variational_problem(self):
        # function spaces
        
        self.gradFEspace = TensorFunctionSpace(self.mesh, "DG", 0) # tensor space to compute the derivative
        self.P0 = FunctionSpace(self.mesh, "DG", 0)
        self.P1 = VectorFunctionSpace(self.mesh, "CG", 1)
        self.mixedFEspace = self.P1 * self.P1
        self.vectorDG0 = VectorFunctionSpace(self.mesh, "DG", 0)

        # self.nullspace = MixedVectorSpaceBasis(
        #     self.mixedFEspace,
        #     [self.mixedFEspace.sub(0), VectorSpaceBasis(constant=True)],
        # )
        
        # self.nullspace = MixedVectorSpaceBasis(
        #     self.mixedFEspace,
        #     [VectorSpaceBasis(constant=True), self.mixedFEspace.sub(1)],
        # )

        X0 = Function(self.P1)
        X1 = Function(self.P1)
        X2 = Function(self.P1)

        X0.interpolate(Constant((1.0, 0.0, 0.0)))
        X1.interpolate(Constant((0.0, 1.0, 0.0)))
        X2.interpolate(Constant((0.0, 0.0, 1.0)))

        X_nullspace = VectorSpaceBasis([X0, X1, X2])
        X_nullspace.orthonormalize()

        self.nullspace = MixedVectorSpaceBasis(
            self.mixedFEspace,
            [
                X_nullspace,
                self.mixedFEspace.sub(1)
            ]
        )

        x, y, z = SpatialCoordinate(self.mesh)
        mu_expr = self.mu_fun(x,y,z)
        lmbda_expr = self.lmbda_fun(x,y,z)
        # interpolate data
        self.mu = Function(self.P0).interpolate(mu_expr)
        self.lmbda = Function(self.P0).interpolate(lmbda_expr)

        volume = assemble(Constant(1.0) * dx(domain=self.mesh))
        mu_avg = assemble(self.mu * dx) / volume
        lmbda_avg = assemble(self.lmbda * dx) / volume

        VTKFile("mu.pvd").write(self.mu)

        i, j, r, t = indices(4)
        kron = Identity(3)
        self.cst = as_tensor(
            self.mu*kron[i,r]*kron[j,t] + self.lmbda*kron[i,j]*kron[r,t],
            (i, j, r, t)
        )
        self.CST = as_tensor(
            mu_avg*kron[i,r]*kron[j,t] + lmbda_avg*kron[i,j]*kron[r,t],
            (i, j, r, t)
        )

        for iii in range(3):
            for jjj in range(3):
                for kkk in range(3):
                    for lll in range(3):
                        self.CST_np[iii,jjj,kkk,lll] = (
                            mu_avg*(iii==kkk)*(jjj==lll)
                            +
                            lmbda_avg*(iii==jjj)*(kkk==lll)
                        )

        # print(type(self.cst))
        # print(self.cst.ufl_shape)
        # print(repr(self.cst))

        # variational forms
        u, r = TrialFunctions(self.mixedFEspace)
        u_test, r_test = TestFunctions(self.mixedFEspace)

        def mskw(A):
            return as_vector([
                A[2,1] - A[1,2],
                A[0,2] - A[2,0],
                A[1,0] - A[0,1]
            ])

        def vskw(v):
            return as_matrix([
                [    0, -v[2],  v[1]],
                [ v[2],     0, -v[0]],
                [-v[1],  v[0],     0]
            ])

        epsilon = grad(u) + vskw(r)
        
        sigma = as_tensor(
            sum(self.cst[i,j,r,t]*epsilon[r,t] for r in range(3) for t in range(3)),
            (i,j)
        )

        # print(type(epsilon))
        # print(epsilon.ufl_shape)
        # # print(repr(epsilon))

        # print(type(self.cst))
        # print(self.cst.ufl_shape)
        # # print(repr(self.cst))

        # print(type(sigma))
        # print(sigma.ufl_shape)
        # # print(repr(sigma))
        
        
        #TODO: check that mskw(vskw) = 2! -- > DONE
        self.a = inner(sigma, grad(u_test))*dx + dot(mskw(epsilon), r_test)*dx

    def _build_solver(self):
        A = assemble(self.a,
                    mat_type="aij")

        self.solver = LinearSolver(
            A,
            nullspace=self.nullspace,
            solver_parameters={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps"
            }
        )
        

    def solve(self, idx, N_idx, c_prev, tildec_prev, C_prev, tildeC_prev):

        def mskew(A):
            return as_vector([
                A[2,1] - A[1,2],
                A[0,2] - A[2,0],
                A[1,0] - A[0,1]
            ])
        def vskew(v):
            return as_matrix([
                [    0, -v[2],  v[1]],
                [ v[2],     0, -v[0]],
                [-v[1],  v[0],     0]
            ])
        
        kron = Identity(3)
        volume = assemble(Constant(1.0)*dx(domain=self.mesh))

        u_test, r_test = TestFunctions(self.mixedFEspace)

        # -----------------------------------------
        # gradU related systems
        # -----------------------------------------        
        ek = as_vector(tuple(kron[:,idx[-1]]))

        i, j, r, t = indices(4)
        prev_term = outer(N_idx, ek)
        extra_term = as_tensor(
            self.cst[i,j,r,t]*prev_term[r,t],
            (i,j)
        )

        rhs_uu = -inner(extra_term, grad(u_test)) * dx(domain=self.mesh)
        rhs_ur = -dot(mskew(outer(N_idx, ek)), r_test) * dx(domain=self.mesh)
        
        if len(idx) >= 3:
            rhs_uu = rhs_uu + dot(c_prev - self.beta * C_prev,u_test) * dx(domain=self.mesh)

        L_u = rhs_uu + rhs_ur
        b = assemble(L_u)
        wh1 = Function(self.mixedFEspace)
        u1, r1 = wh1.subfunctions
        self.solver.solve(wh1, b)

        # compute c and C
        eps = outer(N_idx, ek) + grad(u1) + vskew(r1)
        c = Function(self.gradFEspace)
        c.interpolate(as_tensor(
            self.cst[i,j,r,t]*eps[r,t],
            (i,j)
        ))
        C = np.zeros((3,3))
        for ii in range(3):
            for jj in range(3):
                C[ii,jj] = assemble(c[ii,jj]*dx)/volume

        print("||X|| =", norm(u1))
        print("||P|| =", norm(r1))
        # print(type(c))
        # print(c.ufl_shape)
        # # print(repr(c))
        
        # print(type(C))
        # # print(C.ufl_shape) # ERROR
        # # print(repr(C))


        # -----------------------------------------
        # gradS*R related system
        # -----------------------------------------
        rhs_ru = dot(as_vector(Constant((0.,0.,0.))), u_test) * dx(domain=self.mesh)

        if len(idx) >= 3:
            rhs_ru = rhs_ru + dot(tildec_prev - self.beta* tildeC_prev, u_test) * dx(domain=self.mesh)

        L_r = rhs_ru
        b = assemble(L_r)
        wh2 = Function(self.mixedFEspace)
        self.solver.solve(wh2, b)

        u2, r2 = wh2.subfunctions

        # print("||xi|| =", norm(u2))
        # print("||eta|| =", norm(r2))

        # compute tildec and tildeC
        tilde_eps= grad(u2) + vskew(r2)
        tildec = Function(self.gradFEspace)
        tildec.interpolate(as_tensor(
            self.cst[i,j,r,t]*tilde_eps[r,t],
            (i,j)
        ))

        tildeC = np.zeros((3,3))
        for ii in range(3):
            for jj in range(3):
                tildeC[ii,jj] = assemble(tildec[ii,jj]*dx)/volume

        self.export_solution(
            idx,
            u1, r1,
            u2, r2,
            c, C,
            tildec, tildeC
        )

        return c, tildec, C, tildeC, u1
    

    def export_solution(
        self,
        idx,
        u1, r1,
        u2, r2,
        c, C,
        tildec, tildeC
    ):

        folder = os.path.join("output", f"order_{len(idx)}")
        os.makedirs(folder, exist_ok=True)

        suffix = "".join(map(str, idx))

        u1.rename("X")
        r1.rename("P")

        u2.rename("Xi")
        r2.rename("Eta")

        c.rename("c")
        tildec.rename("tildec")


        VTKFile(os.path.join(folder, f"X_{suffix}.pvd")).write(u1)
        VTKFile(os.path.join(folder, f"P_{suffix}.pvd")).write(r1)

        VTKFile(os.path.join(folder, f"Xi_{suffix}.pvd")).write(u2)
        VTKFile(os.path.join(folder, f"Eta_{suffix}.pvd")).write(r2)

        VTKFile(os.path.join(folder, f"c_{suffix}.pvd")).write(c)
        VTKFile(os.path.join(folder, f"tildec_{suffix}.pvd")).write(tildec)

        Cfun = Function(self.gradFEspace, name="C")
        Cfun.interpolate(as_tensor(C.tolist()))

        tCfun = Function(self.gradFEspace, name="tildeC")
        tCfun.interpolate(as_tensor(tildeC.tolist()))

        VTKFile(os.path.join(folder, f"C_{suffix}.pvd")).write(Cfun)
        VTKFile(os.path.join(folder, f"tildeC_{suffix}.pvd")).write(tCfun)

        np.save(os.path.join(folder, f"C_{suffix}.npy"), C)
        np.save(os.path.join(folder, f"tildeC_{suffix}.npy"), tildeC)
            