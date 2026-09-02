from firedrake import *
from ufl import as_vector, as_matrix
import numpy as np
import os
from firedrake.output import VTKFile
current_path = os.getcwd()
print(current_path)

class CellProblem:
    def __init__(self, n, mu, lmbda, dim, output_dir="output/mixed"):

        print("Initializing cell problem...")

        self.dim = dim
        self.mu_fun = mu
        self.lmbda_fun = lmbda
        self.output_dir = output_dir

        # Mesh
        if dim == 3:
            self.mesh = PeriodicUnitCubeMesh(n, n, n, hexahedral=True)
        elif dim == 2:
            self.mesh = PeriodicUnitSquareMesh(n, n, quadrilateral=True)
        else:
            raise ValueError("dim must be 2 or 3")

        # Function spaces
        self.gradFEspace = TensorFunctionSpace(self.mesh, "DG", 0)
        self.P0 = FunctionSpace(self.mesh, "DG", 0)
        self.P1 = VectorFunctionSpace(self.mesh, "CG", 1)
        
        if self.dim == 3:
            self.Rspace = VectorFunctionSpace(self.mesh,"DG",0)
        else:
            self.Rspace = FunctionSpace(self.mesh,"DG",0)

        self.mixedFEspace = self.P1 * self.Rspace

        # self.vectorDG0 = VectorFunctionSpace(self.mesh, "DG", 0)

        # Material coefficients
        if dim == 3:
            x, y, z = SpatialCoordinate(self.mesh)
        else:
            x, y = SpatialCoordinate(self.mesh)
            z = Constant(0.0)
        
        mu_expr = self.mu_fun(x, y, z)
        lmbda_expr = self.lmbda_fun(x, y, z)
        
        self.mu = Function(self.P0).interpolate(mu_expr)
        self.lmbda = Function(self.P0).interpolate(lmbda_expr)
        
        self.mu.rename("mu")
        self.lmbda.rename("lambda")
        
        os.makedirs(output_dir, exist_ok=True)
        
        VTKFile(os.path.join(output_dir, "mu.pvd")).write(self.mu)
        VTKFile(os.path.join(output_dir, "lambda.pvd")).write(self.lmbda)

        i, j, r, t = indices(4)
        I = Identity(self.dim)
        
        self.cst = as_tensor(
            2.0 * self.mu * I[i, r] * I[j, t]
            +
            self.lmbda * I[i, j] * I[r, t],
            (i, j, r, t)
        )

        # Nullspace
        basis = []
        for i in range(self.dim):
            X = Function(self.P1)
            value = np.zeros(self.dim)
            value[i] = 1
            X.interpolate(Constant(tuple(value)))
            basis.append(X)
        X_nullspace = VectorSpaceBasis(basis)
        X_nullspace.orthonormalize()

        self.nullspace = MixedVectorSpaceBasis(
            self.mixedFEspace,
            [
                X_nullspace,
                self.mixedFEspace.sub(1)
            ]
        )

        # Variational problem
        self._build_variational_problem()

        # Solver
        self._build_solver()

        # Storage for homogenized coefficients
        self.Ceff = {}
        self.tCeff = {} # Optional storage of cell solutions self.solutions = {}
        self.solutions = {}

        print(" done.")


    def _build_variational_problem(self):

        u, r = TrialFunctions(self.mixedFEspace)
        u_test, r_test = TestFunctions(self.mixedFEspace)

        eps = grad(u) + self.vskw(r)

        i, j, rr, tt = indices(4)
        
        sigma = as_tensor(
            sum(self.cst[i,j,rr,t]*eps[rr,t] for rr in range(self.dim) for t in range(self.dim)),
            (i,j)
        )

        self.a = inner(sigma, grad(u_test))*dx + dot(self.mskw(eps), r_test)*dx


    def _build_solver(self):

        print("Assembling...")

        A = assemble(self.a, mat_type="aij")

        print(" done.")

        self.solver = LinearSolver(
            A,
            nullspace=self.nullspace,
            solver_parameters={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps"
            }
        )
        

    def solve(self, l, m):
        """
        Solve the mixed first-order cell problem

            -div[c : (e_l x e_m + grad(X) + vskw(P))] = 0

        together with the mixed constraint

            mskw(e_l x e_m + grad(X) + vskw(P)) = 0.
        
        The resulting effective tensor contribution is
        
            Ceff[:, :, l, m] = <sigma[:, :]>
            
        """
        
        print( f"\nSolving mixed cell problem ({l},{m})" )

        e_l = np.zeros(self.dim)
        e_m = np.zeros(self.dim)
        e_l[l] = 1.0
        e_m[m] = 1.0
        e_l = as_vector(tuple(e_l))
        e_m = as_vector(tuple(e_m))
        A_macro = outer(e_l, e_m)
        A_sym = 0.5*( outer(e_l, e_m) + outer(e_m, e_l))

        i, j, r, t = indices(4)
        
        # -----------------------------------------
        # gradU related systems
        
        # RHS
        # a((X,P),(v,q))
        #     = - integral c:A_sym : grad(v) 
        #
        # --------------------------------------------------
        sigma_macro = as_tensor(sum(
            self.cst[i, j, r, t] * A_sym[r, t]
            for r in range(self.dim)
            for t in range(self.dim)),
            (i, j)
        )

        
        u_test, r_test = TestFunctions( self.mixedFEspace )
        
        L = ( -inner(sigma_macro, grad(u_test)) - dot(Constant(0.), r_test) ) * dx(domain=self.mesh)
        b = assemble(L)

        wh1 = Function(self.mixedFEspace)
        self.solver.solve(wh1, b)
        
        X, P = wh1.subfunctions

        X.rename(f"X_{l}{m}")
        P.rename(f"P_{l}{m}")

        # Microscopic strain and stress
        eps = A_macro + grad(X) + self.vskw(P)

        skweps_expr = self.mskw(grad(X) + self.vskw(P))
        skweps = Function(self.Rspace, name="skweps")
        skweps.interpolate(skweps_expr)
        print("||Seps|| =", norm(skweps))

        P_expected = Function(self.Rspace, name="P_expected")
        P_expected.interpolate(-0.5 * self.mskw(grad(X)))
        P_diff = Function(self.Rspace, name="P_diff")
        P_diff.interpolate(P - P_expected)
        print("||P + 0.5 mskw(gradX)|| =", norm(P_diff))

       
        c_expr = as_tensor(
            sum( self.cst[i, j, r, t] * eps[r, t]
                for r in range(self.dim)
                for t in range(self.dim)),
                (i, j)
            )
        
        c = Function(self.gradFEspace, name = "c")
        c.interpolate(c_expr)

        # Effective tensor
        volume = assemble(Constant(1.0) * dx(domain=self.mesh))

        C = np.zeros((self.dim,self.dim))
        for ii in range(self.dim):
            for jj in range(self.dim):
                C[ii,jj] = assemble(c[ii,jj]*dx)/volume

        rotation_residual = assemble(
            dot(
                self.mskw(grad(X) + self.vskw(P)),
                r_test
            ) * dx
        )

        print("||X|| =", norm(X))
        print("||P|| =", norm(P))
        print("||eps|| = ", norm(eps))
        # print("||Seps|| = ", norm(skweps))
        
        # with rotation_residual.dat.vec_ro as vec:
        #     print("||discrete rotational residual|| =", vec.norm())
        # print("||skw_gradX|| =", norm(self.mskw(grad(X))))
        # print("||skw_P||     =", norm(self.mskw(self.vskw(P))))

        # Export
        folder = os.path.join( self.output_dir, f"cell_{l}{m}" )
        os.makedirs( folder, exist_ok=True )

        VTKFile(os.path.join( folder, f"P_{l}{m}.pvd")).write(P)
        VTKFile(os.path.join( folder, f"sigma_{l}{m}.pvd")).write(c)
        np.save(os.path.join( folder, f"C_{l}{m}.npy"), C)

        return X, P, c, C
    
    def solve_all(self):
        Ceff = np.zeros((self.dim, self.dim, self.dim, self.dim))

        solutions = {}
        for l in range(self.dim):
            for m in range(self.dim):
                X, P, sigma, C_lm = self.solve(l, m)
                solutions[(l, m)] = (X, P, sigma)
                Ceff[:, :, l, m] = C_lm

            self.Ceff = Ceff
            np.save(os.path.join( self.output_dir, "Ceff.npy"), Ceff)

        
        tCeff = np.zeros((self.dim, self.dim, self.dim, self.dim))
        volume = assemble(Constant(1.0) * dx(domain=self.mesh))
        for i in range(self.dim):
            for j in range(self.dim):
                for l in range(self.dim):
                    for m in range(self.dim):
                        tCeff[i, j, l, m] = (
                            assemble( self.cst[i, j, l, m] * dx ) / volume
                        )

        self.tCeff = tCeff

        return Ceff, tCeff, solutions


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

    def symgrad(self, u):

        return 0.5 * (
            grad(u) + grad(u).T
        )