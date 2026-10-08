from firedrake import *
from geometry import generate_periodic_X_mesh, check_gmsh_periodicity, check_periodic_X_mesh, generate_periodic_66_mesh
from ufl import as_tensor, as_matrix
import numpy as np
import os
import inspect

class FineScaleProblem:

    def __init__(self, data, f_fun):
        print("\nInitializing fine scale problem...")
        fs_geom = data["geometry"]["fs_geom"]
        nx = fs_geom["nx_macro"]
        ny = fs_geom["ny_macro"]
        nz = fs_geom["nz_macro"]
        n_micro = fs_geom["n_micro_fs"]
        l_micro = fs_geom["l_micro"]
        
        print(f"n_micro, l_micro = {n_micro}, {l_micro}")

        self.dim = data["dim"]
        self.is_per = data["flags"]["is_per"]

        self.cst = {}
        self.mu_dark = data["coefficients"]["mu_dark"]
        self.mu_light = data["coefficients"]["mu_light"]
        self.lmbda_dark = data["coefficients"]["lmbda_dark"]
        self.lmbda_light = data["coefficients"]["lmbda_light"]
        
        self.f_fun = f_fun
        self.use_gmsh = data["flags"]["gmsh"]

        if not self.is_per:
            if self.dim == 3:
                self.mesh = BoxMesh(
                    nx*n_micro, ny*n_micro, nz*n_micro, nx*l_micro, ny*l_micro, nz*l_micro,
                    hexahedral=True
                )
            elif self.dim == 2:
                self.mesh = RectangleMesh(
                    nx*n_micro, ny*n_micro, nx*l_micro, ny*l_micro,
                    quadrilateral=True
                )
            else:
                raise ValueError("dim must be 2 or 3")
        else:
            if not self.use_gmsh:
                if self.dim == 3:
                    self.mesh = PeriodicBoxMesh(
                        nx*n_micro, ny*n_micro, nz*n_micro, nx*l_micro, ny*l_micro, nz*l_micro,
                        hexahedral=True
                    )
                elif self.dim == 2:
                    self.mesh = PeriodicRectangleMesh(
                        nx*n_micro, ny*n_micro, nx*l_micro, ny*l_micro,
                        quadrilateral=True
                    )
                else:
                    raise ValueError("dim must be 2 or 3")
            else:
                num_conv = data["convergence"]["numerical_refinements"]
                current_level = data["convergence"]["current_level"]

                if data["flags"]["microstructure"] == "X":
                    t=data["geometry"]["crossed"]["t"]
                    # LOOK FOR THE FILE, OTHERWISE GENERATE
                    filename = f"output/Xfs_mesh_level{current_level}.msh"
                    
                    # If the file does not exist, generate it (and all the required nested refinemnets)
                    if not os.path.isfile(filename):
                        generate_periodic_X_mesh(
                            "output/Xfs_mesh.msh",
                            nx, ny, l_micro,
                            t,
                            mesh_size_matrix=4*t/n_micro,
                            mesh_size_fiber=t/n_micro,
                            num_refinements = num_conv
                        )
                elif data["flags"]["microstructure"] == "66":
                    l = data["geometry"]["chiral"]["l"]
                    R = data["geometry"]["chiral"]["R"]
                    t = data["geometry"]["chiral"]["t"]
                    
                    filename = f"output/66fs_mesh_level{current_level}.msh"

                    if not os.path.isfile(filename):
                        generate_periodic_66_mesh(
                            "output/66fs_mesh.msh",
                            nx, ny, l_micro,
                            l, R, t,
                            mesh_size_matrix=2*t/n_micro,
                            mesh_size_fiber=t/n_micro,
                            mesh_size_interface=t/n_micro,
                            transition_width=t/n_micro,
                            num_refinements = num_conv
                        )


                # check_gmsh_periodicity(filename)
                # Tell PETSc/DMPlex to read the $Periodic section
                # from the Gmsh file.
                opts = PETSc.Options()
                opts["dm_plex_gmsh_periodic"] = True
                opts["dm_plex_gmsh_use_regions"] = True
                opts["dm_plex_gmsh_use_generic"] = True

                self.mesh = Mesh(filename)

                # check_periodic_X_mesh(self.mesh)

        self._build_spaces()
        self._build_variational_problem()
        self._build_solver()
        print(" done.")


    def _build_spaces(self):
        self.P0 = FunctionSpace(self.mesh, "DG", 0)
        self.P1 = VectorFunctionSpace(self.mesh, "CG", 1)
        self.Uspace = VectorFunctionSpace(self.mesh, "CG", 2)

    def _build_variational_problem(self):
        # data
        if self.dim == 3:
            x, y, z = SpatialCoordinate(self.mesh)
        else:
            x,y = SpatialCoordinate(self.mesh)
            z = Constant(0.)

        # mu_expr = self.mu_fun(x,y,z)
        # lmbda_expr = self.lmbda_fun(x,y,z)

        # self.mu = Function(self.P0).interpolate(mu_expr)
        # self.lmbda = Function(self.P0).interpolate(lmbda_expr)

        
        fiber_cells = self.mesh.cell_subset(1)
        matrix_cells = self.mesh.cell_subset(2)

        self.mu = Function(self.P0, name="mu")
        self.lmbda = Function(self.P0, name="lambda")

        self.mu.dat.data[:] = Constant(self.mu_light)
        self.lmbda.dat.data[:] = Constant(self.lmbda_light)

        fiber_cells = self.mesh.cell_subset(1)

        self.mu.dat.data[fiber_cells.indices] = Constant(self.mu_dark)
        self.lmbda.dat.data[fiber_cells.indices] = Constant(self.lmbda_dark)

        self.mu.rename("mu")
        self.lmbda.rename("lmbda")

        I = Identity(self.dim)
        
        i, j, r, t = indices(4)

        self.cst = as_tensor(
            self.mu * (I[i,r]*I[j,t] + I[i,t]*I[j,r])
            + self.lmbda * I[i,j]*I[r,t],
            (i, j, r, t)
        )

        # Nullspace
        if self.is_per:
            basis = []
            for ii in range(self.dim):
                vv = Function(self.Uspace)
                value = np.zeros(self.dim)
                value[ii] = 1.0
                vv.interpolate(Constant(tuple(value)))
                basis.append(vv)
            self.nullspace = VectorSpaceBasis(basis)
            self.nullspace.orthonormalize()

        # Variational problem
        u = TrialFunction(self.Uspace)
        v = TestFunction(self.Uspace)

        strain = self.symgrad(u)

        i, j, r, t = indices(4)
        sigma = as_tensor(
            sum(
                self.cst[i, j, r, t] * strain[r, t]
                for r in range(self.dim)
                for t in range(self.dim)
            ),
            (i, j)
        )

        self.a = inner(sigma, self.symgrad(v)) * dx

    def _build_solver(self):

        if not self.is_per:
            self.bc_Dir = DirichletBC(
                self.mixed.sub(0),
                Constant(np.zeros(self.dim)),
                1
            )

            bcs=self.bc_Dir

        else:
            bcs = None

        print("Assembling...")
        A = assemble(
            self.a,
            mat_type="aij",
            bcs=bcs,
        )
        print(" done.")

        params = {
            "ksp_type": "cg",
            "pc_type": "hypre",
            "pc_hypre_type": "boomeramg",
            "ksp_rtol": 1e-8,
            "ksp_atol": 1e-12,
            "ksp_max_it": 1000,
            "ksp_view": None,
            "ksp_converged_reason": None,
        }
        
        params={
            "ksp_type":"preonly",
            "pc_type":"lu",
            "pc_factor_mat_solver_type":"mumps"
        }

        nsp = self.nullspace if self.is_per else None
        self.solver = LinearSolver(
            A,
            nullspace=nsp,
            solver_parameters = params
        )

    def solve(self):

        V = TestFunction(self.Uspace)
        if self.dim == 3:
            x, y, z = SpatialCoordinate(self.mesh)
        else:
            x, y = SpatialCoordinate(self.mesh)
            z = Constant(0.)

        f_expr = self.f_fun(x,y,z)
        force = Function(self.Uspace).interpolate(f_expr)

        # L = dot(force,V)*ds(2)
        L = dot(force,V)*dx
        b = assemble(L)

        U = Function(self.Uspace)

        print("Solving fine scale problem...")
        self.solver.solve(U, b)
        print(" done.")

        self.export_solution(U)
    
        return U

    def export_solution(self, U):
        folder = os.path.join("output/finescale")
        os.makedirs(folder, exist_ok=True)

        U.rename("U_fs")

        VTKFile(os.path.join(folder, f"U_fs.pvd")).write(U)        
        VTKFile(os.path.join(folder, f"mu_fs.pvd")).write(self.mu)
        VTKFile(os.path.join(folder, f"lambda_fs.pvd")).write(self.lmbda)


    def _dict_to_tensor4(self, Cdict):
        C = np.zeros((self.dim,self.dim,self.dim,self.dim))

        for (l, m), C_lm in Cdict.items():
            C[:, :, l, m] = np.asarray(C_lm)

        return C

    def symgrad(self, u):
        return 0.5 * (
            grad(u) + grad(u).T
        )

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