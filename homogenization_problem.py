from firedrake import *
from cell_problem import CellProblem
from itertools import product, combinations_with_replacement
import numpy as np
import os

class HomogenizationProblem:
    def __init__(self, cell_problem, max_order):
        self.cell = cell_problem
        self.max_order = max_order
        self.store = SolutionStore(cell_problem.mixedFEspace, cell_problem.gradFEspace)

    def run(self):
        for l in range(0, self.max_order + 1):
            self.solve_order(l)

    def solve_order(self, l):
        print(f"\nORDER {l}:")
        indices = self._generate_multiindices(l)        
        for multi_idx in indices:
            idx_str = "(" + ",".join(map(str, multi_idx)) + ")"
            print(f"  multi-index = {idx_str}")

            previous = None if l == 0 else multi_idx[:-1]
            pprevious = None if (l == 0 or l == 1) else multi_idx[:-2]
            for a in range(3):
                print(f"    a = {a}")

                r0 = self.store.get_r(previous, a)
                u00 = self.store.get_u(pprevious, a)
                u0 = self.store.get_u(previous, a)
                du0 = self.store.get_du(previous, a)
                dmuu0 = self.store.get_dmuu(previous, a)

                r, u, du, dmuu, LL = self.cell.solve(multi_idx, a, r0, u0, u00, du0, dmuu0)

                self.store.save(multi_idx, a, r, u, du, dmuu, LL)
    
        # self.compute_effective_tensor(sol)
    
    def _generate_multiindices(self, l):
        if l == 0:
            return [()]
        return [
            tuple(idx)
            for idx in combinations_with_replacement(range(3), l)
        ]
    

class SolutionStore:
    def __init__(self, mixedFEspace, gradFEspace):
        self.r = {}
        self.u = {}
        self.du = {}
        self.dmuu = {}
        self.LL = {}
        self.zero_u, self.zero_r  = Function(mixedFEspace).subfunctions
        self.zero_du = Function(gradFEspace).interpolate(grad(self.zero_u))

    def save(self, multiindex, a, r, u, du, dmuu, LL):
        key = (tuple(multiindex), a)
        self.r[key] = (r)
        self.u[key] = (u)
        self.du[key] = (du)
        self.dmuu[key] = (dmuu)
        self.LL[key] = (LL)

    def get_r(self, multiindex, a):
        return self.zero_r if multiindex == None else self.r[(tuple(multiindex), a)]
    
    def get_u(self, multiindex, a):
        return self.zero_u if multiindex == None else self.u[(tuple(multiindex), a)]
    
    def get_du(self, multiindex, a):
        return self.zero_du if multiindex == None else self.du[(tuple(multiindex), a)]
    
    def get_dmuu(self, multiindex, a):
        return self.zero_du if multiindex == None else self.dmuu[(tuple(multiindex), a)]


    # def has(self, multiindex):
    #     return tuple(multiindex) in self.data
