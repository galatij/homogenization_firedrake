from firedrake import *
from ufl import as_tensor, as_matrix
import numpy as np
import os


class HoEffectiveElasticityProblem:

    def __init__(self, mesh, Ceff, tCeff, f, dim, order):
        self.mesh = mesh
        self.dim = dim
        self.order = order
        self.Ceff = Ceff
        self.tCeff = tCeff
        self.f_fun = f

        self._build_spaces()
        self._build_variational_problem()
        self._build_solver()


    def _build_spaces(self):

        self.Uspace = VectorFunctionSpace(self.mesh, "CG", 2)

        if self.dim == 3:
            self.Rspace = VectorFunctionSpace(self.mesh, "CG", 1)
        else:
            self.Rspace = FunctionSpace(self.mesh, "CG", 1)

        spaces = [self.Uspace, self.Rspace]

        self.Gspaces = [] # spaces for ho gradU
        self.Hspaces = [] # spaces for ho gradR

        # Auxiliary variables for higher gradients of U
        # G1 = grad(U), G2 = grad(G1), ..., G_{order-2}
        rank = 2 #################### (rk gradU)

        for k in range(1, self.order - 1):
            shape = tuple([self.dim] * rank)
            V = TensorFunctionSpace(self.mesh, "CG", 1, shape=shape)

            self.Gspaces.append(V)
            spaces.append(V)

            rank += 1

        # Auxiliary variables for higher gradients of R
        # H1 = grad(R), H2 = grad(H1), ...
        # if self.dim == 3:
        rank = 2
        for k in range(1, self.order-1):
            shape = tuple([self.dim] * rank)
            V = TensorFunctionSpace(self.mesh, "DG", 0, shape=shape)

            self.Hspaces.append(V)
            spaces.append(V)

            rank += 1
        # else:  # R is scalar in 2D: H1 is a vector, H2 is a matrix, H3 is rank-3, ...
        #     rank = 1
        #     for k in range(1, self.order-1):
        #         if rank == 1:
        #             V = VectorFunctionSpace(self.mesh, "DG", 0)
        #         else:
        #             shape = tuple(self.dim for _ in range(rank))
        #             V = TensorFunctionSpace(self.mesh, "DG", 0, shape=shape)

        #         self.Hspaces.append(V)
        #         spaces.append(V)

        #         rank += 1

        self.mixed = MixedFunctionSpace(tuple(spaces))



    def _build_variational_problem(self):
        trial = TrialFunctions(self.mixed)
        test = TestFunctions(self.mixed)

        U = trial[0]
        R = trial[1]

        V = test[0]
        Q = test[1]

        nG = len(self.Gspaces)
        nH = len(self.Hspaces)

        G = list(trial[2:2+nG])
        H = list(trial[2+nG:])

        VG = list(test[2:2+nG])
        VH = list(test[2+nG:])

        eps = grad(U) + self.vskw(R)

        i,j,r,t = indices(4)

        # effective stress
        C0 = as_tensor(self._dict_to_tensor4(self.Ceff[2]))
        tC0 = as_tensor(self._dict_to_tensor4(self.tCeff[2]))

        Sigma = as_tensor(
            C0[i,j,r,t]*grad(U)[r,t] + tC0[i,j,r,t]*self.vskw(R)[r,t],
            (i,j)
        )

        for k in range(1, nG):
            order = k+2

            Ck = self.dict_to_np_tensor(self.Ceff[order]) 
            tCk = self.dict_to_np_tensor(self.tCeff[order])
            # print("order =", order)
            # print("C shape =", Ck.shape)
            # print("G shape =", G[k].ufl_shape)
            Sigma += self.contract(Ck, G[k]) + self.contract(tCk, H[k])

            
        # bilinear form
        self.V = V
        self.Q = Q

        self.a = (inner(Sigma,grad(V)) + dot(self.mskw(eps),Q)) * dx

        # weak constraints for the additional variables
        
        self.a += inner(G[0] - grad(U),VG[0]) * dx + inner(H[0] - self.vskw(R), VH[0]) * dx
        for k in range(1, nG):
            self.a += inner(G[k] - grad(G[k-1]),VG[k]) * dx + inner(H[k] - grad(H[k-1]),VH[k]) * dx

        if self.dim == 3:
            x, y, z = SpatialCoordinate(self.mesh)
        else:
            x, y = SpatialCoordinate(self.mesh)
            z = Constant(0.)
        
        f_expr = self.f_fun(x,y,z)
        force = Function(self.Uspace).interpolate(f_expr)
        
        L = dot(force,V)*dx
        self.b = assemble(L)


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

        w = Function(self.mixed)

        self.solver.solve(w, self.b)

        U= w.subfunctions[0]
        R= w.subfunctions[1]

        self.export_solution(U, R)
    
        return U,R

    def export_solution(
        self,
        U, R
    ):
        folder = os.path.join("output/hoeffective")
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

    def contract(self, Ck_np, Gk):
        """
        Contract a homogenized tensor C^(k) with a rank-k tensor G.

        Parameters
        ----------
        Ck_np : numpy.ndarray
            Shape (dim, dim, dim, ..., dim)
            First two indices are stress indices.

        Gk : UFL tensor
            Rank k tensor.

        Returns
        -------
        UFL tensor
            Rank 2 tensor Sigma_ij
        """

        Ck = as_tensor(Ck_np)
        rank_C = len(Ck_np.shape)
        rank_G = len(Gk.ufl_shape)

        # number of derivative indices
        k = rank_C - 2

        assert(k==rank_G)

        ind = indices(rank_C)

        # full tensor component C_{ijlmn...}
        expr = Ck[tuple(ind)]

        # contract with Gk
        expr *= Gk[tuple(ind[2:])]

        # resulting rank-2 tensor
        Sigma = as_tensor(expr, (ind[0], ind[1]))

        return Sigma

    def dict_to_np_tensor(self, Ck):
        # Ck: dictionary with key = multi-index, value = rank 2 UFL tensor
        key0 = next(iter(Ck))
        n = len(key0)

        C = np.zeros((self.dim,self.dim) + (self.dim,)*n)

        for key, A in Ck.items():
            C[(slice(None),slice(None))+key] = A

        return C