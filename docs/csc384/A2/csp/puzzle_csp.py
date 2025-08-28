#Look for #IMPLEMENT tags in this file.
'''
All encodings need to return a CSP object, and a list of lists of Variable objects 
representing the board. The returned list of lists is used to access the 
solution. 

For example, after these three lines of code

    csp, var_array = caged_csp(board)
    solver = BT(csp)
    solver.bt_search(prop_FC, var_ord)

var_array[0][0].get_assigned_value() should be the correct value in the top left
cell of the FunPuzz puzzle.

The grid-only encodings do not need to encode the cage constraints.

1. binary_ne_grid (worth 10/100 marks)
    - An enconding of a FunPuzz grid (without cage constraints) built using only 
      binary not-equal constraints for both the row and column constraints.

2. nary_ad_grid (worth 10/100 marks)
    - An enconding of a FunPuzz grid (without cage constraints) built using only n-ary 
      all-different constraints for both the row and column constraints. 

3. caged_csp (worth 25/100 marks) 
    - An enconding built using your choice of (1) binary binary not-equal, or (2) 
      n-ary all-different constraints for the grid.
    - Together with FunPuzz cage constraints.

'''

from enum import Enum, IntEnum
from itertools import product, permutations
from math import prod
from cspbase import CSP, Variable, Constraint

# --------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------
class Op(IntEnum):
    ADD = 0
    SUB = 1
    DIV = 2
    MUL = 3
    NONE = 4          # single-cell cage


class Pos:
    """11 → (0,0), 26 → (1,5) etc."""
    __slots__ = ("r", "c")

    def __init__(self, rc: int):
        self.r = rc // 10 - 1
        self.c = rc % 10 - 1


class Cage:
    """Stores cells, target, op + builds the satisfying-tuple list."""

    def __init__(self, raw: list[int]):
        if len(raw) == 2:
            self.cells = [Pos(raw[0])]
            self.target = raw[1]
            self.op = Op.NONE
        else:
            *cell_ids, self.target, op_code = raw
            self.cells = [Pos(cid) for cid in cell_ids]
            self.op = Op(op_code)

    # ----------------------------------------------------------------
    # build all satisfying tuples, allowing repeated digits
    # ----------------------------------------------------------------

    def sat_tuples(self, domain):
        """Return **all** value tuples (allowing repeats) that satisfy this cage."""
        k = len(self.cells)

        # single-cell cage
        if self.op is Op.NONE:
            return [(self.target,)]

        sats = []

        for vals in product(domain, repeat=k):          # REPEATS ALLOWED
            if self.op is Op.ADD and sum(vals) == self.target:
                sats.append(vals)

            elif self.op is Op.MUL and prod(vals) == self.target:
                sats.append(vals)


            elif self.op is Op.SUB:
                for perm in permutations(vals):
                    res = perm[0]
                    for x in perm[1:]:
                        res -= x
                    if abs(res) == self.target:
                        sats.append(vals)          # keep <vals>, not <perm>
                        break                      # done with this vals

            elif self.op is Op.DIV:
                for perm in permutations(vals):
                    res, ok = perm[0], True
                    for x in perm[1:]:
                        if x == 0 or res % x:
                            ok = False
                            break
                        res //= x
                    if ok and res == self.target:
                        sats.append(vals)          # keep <vals>, not <perm>
                        break

        return sats


# --------------------------------------------------------------------
# grid encoders
# --------------------------------------------------------------------

def binary_ne_grid(fpuzz_grid):
    """
    Return (csp, grid) where each row/column is enforced with
    |row|·(n−1)/2 + |col|·(n−1)/2   binary not-equal constraints.
    """
    n = fpuzz_grid[0][0]
    dom = list(range(1, n + 1))

    grid = [[Variable(f"R{r+1}C{c+1}", dom) for c in range(n)]
            for r in range(n)]
    csp = CSP("Grid_NE", [v for row in grid for v in row])

    ne_pairs = [(a, b) for a in dom for b in dom if a != b]

    # rows
    for r in range(n):
        for c1 in range(n):
            for c2 in range(c1 + 1, n):
                v1, v2 = grid[r][c1], grid[r][c2]
                con = Constraint(f"R{r+1}_C{c1+1}_C{c2+1}", [v1, v2])
                con.add_satisfying_tuples(ne_pairs)
                csp.add_constraint(con)

    # columns
    for c in range(n):
        for r1 in range(n):
            for r2 in range(r1 + 1, n):
                v1, v2 = grid[r1][c], grid[r2][c]
                con = Constraint(f"C{c+1}_R{r1+1}_R{r2+1}", [v1, v2])
                con.add_satisfying_tuples(ne_pairs)
                csp.add_constraint(con)

    return csp, grid

def nary_ad_grid(fpuzz_grid):
    n   = fpuzz_grid[0][0]
    dom = list(range(1, n + 1))

    grid = [[Variable(f"R{r+1}C{c+1}", dom) for c in range(n)]
            for r in range(n)]
    csp = CSP("Grid_AD", [v for row in grid for v in row])

    # all-different tuples once
    all_diff = list(permutations(dom, n))

    # rows
    for r in range(n):
        con = Constraint(f"Row{r+1}", grid[r])
        con.add_satisfying_tuples(all_diff)
        csp.add_constraint(con)

    # columns
    for c in range(n):
        col_vars = [grid[r][c] for r in range(n)]
        con = Constraint(f"Col{c+1}", col_vars)
        con.add_satisfying_tuples(all_diff)
        csp.add_constraint(con)

    return csp, grid


def caged_csp(fpuzz_grid):

    csp, grid = binary_ne_grid(fpuzz_grid)

    n   = fpuzz_grid[0][0]
    dom = list(range(1, n + 1))

    for raw in fpuzz_grid[1:]:
        cage  = Cage(raw)
        scope = [grid[p.r][p.c] for p in cage.cells]

        tuples = cage.sat_tuples(dom)
        con    = Constraint(f"Cage_{cage.op.name}_{cage.target}", scope)
        con.add_satisfying_tuples(tuples)
        csp.add_constraint(con)

    return csp, grid
