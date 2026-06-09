from firedrake import *
from ufl import as_vector, as_matrix
import numpy as np
import os
from firedrake.output import VTKFile
current_path = os.getcwd()
print(current_path)

class CellProblem:
    def __init__(self, n, mu, force):
        self.mesh = PeriodicUnitCubeMesh(n, n, n, hexahedral=False)
        self.mu_fun = mu
        self.force_fun = force

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


        self.nullspace = MixedVectorSpaceBasis(
            self.mixedFEspace,
            [self.mixedFEspace.sub(0), VectorSpaceBasis(constant=True)],
        )

        x, y, z = SpatialCoordinate(self.mesh)
        mu_expr = self.mu_fun(x,y,z)
        # interpolate data
        self.mu = Function(self.P0).interpolate(mu_expr)

        # self.f = Function(self.P1).interpolate(self.force_fun)

        # variational forms
        u, r = TrialFunctions(self.mixedFEspace)
        u_test, r_test = TestFunctions(self.mixedFEspace)

        def mskew(A):
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

        self.a = (self.mu * inner(grad(u), grad(u_test)) + dot(mskew(grad(r)), u_test) + dot(mskew(grad(u)), r_test) + 2 / self.mu * dot(r, r_test))*dx

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
        
        # self.A = assemble(self.a, mat_type="aij")
        # self.ksp = PETSc.KSP().create()
        # self.ksp.setOperators(self.A.M.handle)
        # self.ksp.setType("preonly")
        # self.ksp.getPC().setType("lu")
        # self.ksp.getPC().setFactorSolverType("mumps")
        # self.ksp.setUp()

    def solve(self, idx, a, r0, u0, u00, du0, dmuu0):
        # def cofun2numpy(b):
        #     return b.vector().array()
        
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
        
        kron = np.eye(3)
        ea = [0.0, 0.0, 0.0]
        ea[a] = 1.0
        ea = as_vector(Constant(ea))
        volume = assemble(Constant(1.0)*dx(domain=self.mesh))

        u_test, r_test = TestFunctions(self.mixedFEspace)

        # FIRST SYSTEM
        rhs_uu_trial = as_vector(Constant((0.,0.,0.)))
        rhs_ur_trial = as_vector(Constant((0.,0.,0.)))

        if len(idx) == 1:
            rhs_uu_trial = rhs_uu_trial + self.mu * du0[0][:,idx[-1]] + dmuu0[0][:,idx[-1]] + vskew(r0[0])[:,idx[-1]]
            rhs_ur_trial = rhs_ur_trial + vskew(u0[0])[:,idx[-1]]
        
        elif len(idx) == 2:
            rhs_uu_trial = rhs_uu_trial + self.mu * kron[idx[-1],idx[-2]] * u00[0]

        h_u = as_vector([-assemble(rhs_uu_trial[i] * dx(domain=self.mesh)) / volume for i in range(3)])
        g_u = as_vector([-assemble(rhs_ur_trial[i] * dx(domain=self.mesh)) / volume for i in range(3)])
        # Remark: for l==0 --> h_u = g_u = 0;
        #         for l==1 --> h_u = 0, g_u = - mskw e_{k_l}^T
        #         for l>=2 --> h_u != 0, g_u =? --> impose = 0??
        if len(idx) >= 2:
            g_u = as_vector(Constant((0.,0.,0.)))

        L_u = dot(rhs_uu_trial + h_u, u_test) * dx(domain=self.mesh) + dot(rhs_ur_trial + g_u, r_test) * dx(domain=self.mesh)
        # b = cofun2numpy(assemble(L_u))
        b = assemble(L_u)
        wh1 = Function(self.mixedFEspace)
        u1, r1 = wh1.subfunctions
        if len(idx) == 0:
            u1.assign(Constant(tuple(ea)))
            r1.assign(Constant((0.0, 0.0, 0.0)))
        else:
            self.solver.solve(wh1, b)
            # self.ksp.solve(
            #     b.dat.vec,
            #     wh1.dat.vec
            # )

        du1 = Function(self.gradFEspace).interpolate(grad(u1))
        dmuu1 = Function(self.gradFEspace).interpolate(grad(self.mu*u1))

        # SECOND SYSTEM
        rhs_ru_trial = as_vector(Constant((0.,0.,0.)))
        rhs_rr_trial = as_vector(Constant((0.,0.,0.)))

        if len(idx) == 1:
            rhs_ru_trial = rhs_ru_trial + self.mu * du0[1][:,idx[-1]] + dmuu0[1][:,idx[-1]] + vskew(r0[1])[:,idx[-1]]
            rhs_rr_trial = rhs_rr_trial + vskew(u0[1])[:,idx[-1]]
        
        elif len(idx) == 2:
            rhs_ru_trial = rhs_ru_trial + self.mu * kron[idx[-1],idx[-2]] * u00[1]

        h_r = as_vector([-assemble(rhs_ru_trial[i] * dx(domain=self.mesh)) / volume for i in range(3)])        
        g_r = as_vector([-assemble(rhs_rr_trial[i] * dx(domain=self.mesh)) / volume for i in range(3)])
        # Remark: for l==0 --> h_r = 0, g_r =? --> impose g_r;
        #         for l==1 --> h_r != 0, g_r = 0
        #         for l>=2 --> h_r != 0, g_r =? --> impose = 0??
        if len(idx) == 0:
            g_r = as_vector([-assemble(self.mu * ea[i] * dx(domain=self.mesh)) / volume for i in range(3)])
        elif len(idx) >= 1:
            g_r = as_vector(Constant((0.,0.,0.)))

        L_r = dot(rhs_ru_trial + h_r, u_test) * dx(domain=self.mesh) + dot(rhs_rr_trial + g_r, r_test) * dx(domain=self.mesh)
        # b = cofun2numpy(assemble(L_r))
        b = assemble(L_r)
        wh2 = Function(self.mixedFEspace)
        self.solver.solve(wh2, b)
        # self.ksp.solve(
        #     b.dat.vec,
        #     wh2.dat.vec
        # )

        u2, r2 = wh2.subfunctions
        du2 = Function(self.gradFEspace).interpolate(grad(u2))
        dmuu2 = Function(self.gradFEspace).interpolate(grad(self.mu*u2))

        hu_fun = Function(self.vectorDG0, name="h_u")
        hu_fun.interpolate(h_u)
        gu_fun = Function(self.vectorDG0, name="g_u")
        gu_fun.interpolate(g_u)
        hr_fun = Function(self.vectorDG0, name="h_r")
        hr_fun.interpolate(h_r)
        gr_fun = Function(self.vectorDG0, name="g_r")
        gr_fun.interpolate(g_r)
        self.export_solution(idx, a, r1, r2, u1, u2, du1, du2, dmuu1, dmuu2, hu_fun, gu_fun, hr_fun, gr_fun)

        return (r1, r2), (u1,u2), (du1, du2), (dmuu1, dmuu2), (h_u, g_u, h_r, g_r)

    def export_solution(self, idx, a,
                r1, r2,
                u1, u2,
                du1, du2,
                dmuu1, dmuu2,
                h_u, g_u, h_r, g_r):
        l = len(idx)

        folder = os.path.join("output/correctors", str(l))
        folder_constants = os.path.join("output/constants", str(l))
        os.makedirs(folder, exist_ok=True)
        if l == 0:
            suffix = f"a{a}"
        else:
            suffix = "".join(map(str, idx))
            suffix = f"{suffix}_a{a}"
        
        VTKFile(os.path.join(folder, f"u1_{suffix}.pvd")).write(u1)
        VTKFile(os.path.join(folder, f"u2_{suffix}.pvd")).write(u2)

        VTKFile(os.path.join(folder, f"r1_{suffix}.pvd")).write(r1)
        VTKFile(os.path.join(folder, f"r2_{suffix}.pvd")).write(r2)

        VTKFile(os.path.join(folder, f"du1_{suffix}.pvd")).write(du1)
        VTKFile(os.path.join(folder, f"du2_{suffix}.pvd")).write(du2)

        VTKFile(os.path.join(folder, f"dmuu1_{suffix}.pvd")).write(dmuu1)
        VTKFile(os.path.join(folder, f"dmuu2_{suffix}.pvd")).write(dmuu2)

        VTKFile(os.path.join(folder_constants, f"huu_a{a}.pvd")).write(h_u)        
        VTKFile(os.path.join(folder_constants, f"hru_a{a}.pvd")).write(g_u)
        VTKFile(os.path.join(folder_constants, f"hur_a{a}.pvd")).write(h_r)        
        VTKFile(os.path.join(folder_constants, f"hrr_a{a}.pvd")).write(g_r)

        