from firedrake import *
from cell_problem import CellProblem
from itertools import product, combinations_with_replacement
import numpy as np
import os

class HomogenizationProblem:
    def __init__(self, cell_problem, max_order):
        self.cell = cell_problem
        self.max_order = max_order

        zero_tensor = Function(cell_problem.gradFEspace, name="zero_tensor")
        zero_tensor.interpolate(as_tensor((np.zeros((cell_problem.dim, cell_problem.dim)))))

        self.store = SolutionStore(cell_problem.cst, cell_problem.CST, zero_tensor, cell_problem.dim)
        self.Ceff = {}
        self.tCeff = {}
        for l in range(2, max_order + 1):
            self.Ceff[l] = {}
            self.tCeff[l] = {}

    def run(self):
        for l in range(2, self.max_order + 1):
            self.solve_order(l)

        return self.Ceff, self.tCeff

    def solve_order(self, l):
        print(f"\nORDER {l}:")
        indices = self._generate_multiindices(l)        
        for multi_idx in indices:
            idx_str = "(" + ",".join(map(str, multi_idx)) + ")"
            print(f"  multi-index = {idx_str}")

            key = tuple(multi_idx)
            c_prev = self.store.get_c(key, vec=True)
            C_prev = self.store.get_C(key, vec=True)
            tc_prev = self.store.get_tc(key, vec=True)
            tC_prev = self.store.get_tC(key, vec=True)
            N = self.store.get_N(key, prev=True)

            c, tc, C, tC, u1 = self.cell.solve(
                multi_idx, N, c_prev, tc_prev, C_prev, tC_prev)

            self.store.save(multi_idx, c, tc, C, tC, u1)

            #  # store C0
            # if l == 2:
            #     lm = tuple(multi_idx)
            #     self.Ceff0_np[:, :, lm[0], lm[1]] = C

            key = tuple(multi_idx)
            self.Ceff[l][key] = C.copy()
            self.tCeff[l][key] = tC.copy()

    def _generate_multiindices(self, l):
        if l == 0:
            return [()]

        first_indices = range(self.cell.dim)

        derivative_indices = list(
            combinations_with_replacement(range(self.cell.dim), l - 1)
        )

        return [
            (i, *deriv)
            for i in first_indices
            for deriv in derivative_indices
        ]

class SolutionStore:
    def __init__(self, cst, CST, zero_tensor, dim):
        self.dim = dim
        self.cst = cst
        self.CST = CST
        self.c = {}
        self.tc = {}
        self.C = {}
        self.tC = {}
        self.u = {}
        self.r = {}
        self.zero_tensor = zero_tensor

    def save(self, multiindex, c, tc, C, tC, u):
        key = tuple(multiindex)
        self.c[key] = c
        self.tc[key] = tc
        self.C[key] = C
        self.tC[key] = tC
        self.u[key] = u

    def get_c(self, key, vec=False):
        if not vec:
            return self.c[key]
        # elif vec:
        if len(key) == 2:
            return self.zero_tensor[:, key[-1]]
        #elif len >=3:
        prev_key = key[:-1]
        return self.__tensor_column(self.c[prev_key], key[-1])
    
    def get_tc(self, key, vec=False):
        if not vec:
            return self.tc[key]
        # elif vec:
        if len(key) == 2:
            return self.zero_tensor[:, key[-1]]
        
        prev_key = key[:-1]
        if len(key) == 3:
            return self.__tensor4_column(self.cst, key[-1], key[0], key[1])
        
        #elif len >=4:
        return self.__tensor_column(self.tc[prev_key],key[-1])
    
    
    def get_C(self, key, vec=False):
        if not vec:
            return self.C[key]
        # elif vec:
        if len(key) == 2:
            return as_vector(tuple(np.zeros(self.dim)))
        #elif len >=3:
        prev_key = key[:-1]
        return self.__tensor_column(self.C[prev_key], key[-1])
    

    def get_tC(self, key, vec=False):
        if not vec:
            return self.tC[key]
        # elif vec:
        if len(key) == 2:
            return as_vector(tuple(np.zeros(self.dim)))
        
        prev_key = key[:-1]
        if len(key) == 3:
            return self.__tensor4_column(self.CST, key[-1], key[0], key[1])
        
        #elif len >=4:
        return self.__tensor_column(self.tC[prev_key], key[-1])
    
    def get_N(self, key, prev=False):
        if not prev:
            return self.u[key]
        
        if len(key) == 2:
            kron = Identity(self.dim)
            return as_vector(tuple(kron[:, key[0]]))
        prev_key = key[:-1]
        return self.u[prev_key]
    
    def __tensor_column(self, A, j):
        return as_vector([A[i, j] for i in range(self.dim)])
    
    def __tensor4_column(self, A, j, k, l):
        return as_vector([A[i,j,k,l] for i in range(self.dim)])

        
    # def _generate_multiindices(self, l):
    #     if l == 0:
    #         return [()]
    #     return [
    #         tuple(idx)
    #         for idx in combinations_with_replacement(range(self.cell.dim), l)
    #     ]