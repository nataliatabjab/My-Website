from cspbase import Variable, Constraint, CSP
from propagators import prop_FC, prop_GAC

def test_empty_domain_after_fc():
    print("=== FC Dead-End Detection ===")
    v1 = Variable("V1", [1, 2])
    v2 = Variable("V2", [1, 2])
    c = Constraint("C1", [v1, v2])
    c.add_satisfying_tuples([(1, 2)])

    csp = CSP("FC_Test", [v1, v2])
    csp.add_constraint(c)

    v1.assign(1)
    status, pruned = prop_FC(csp, newVar=v1)

    expected_status = True  # <-- update this
    expected_domain = [2]   # <-- expected domain

    passed = (status == expected_status) and (v2.cur_domain() == expected_domain)

    print("Status:", status)
    print("V2 domain:", v2.cur_domain())
    print("PASS ✅" if passed else "FAIL ❌")
    print()



def test_gac_multiple_constraints():
    print("=== GAC with Multiple Constraints ===")
    x = Variable("X", [1, 2, 3])
    y = Variable("Y", [1, 2, 3])
    z = Variable("Z", [1, 2, 3])

    c1 = Constraint("C1", [x, y])
    c1.add_satisfying_tuples([(1, 2), (2, 3), (3, 1)])

    c2 = Constraint("C2", [y, z])
    c2.add_satisfying_tuples([(2, 2), (3, 3)])

    csp = CSP("GAC_Test", [x, y, z])
    csp.add_constraint(c1)
    csp.add_constraint(c2)

    x.assign(1)
    status, pruned = prop_GAC(csp, newVar=x)

    expected_domains = [[1], [2], [2]]
    actual_domains = [x.cur_domain(), y.cur_domain(), z.cur_domain()]
    passed = actual_domains == expected_domains

    print("Domains:", actual_domains)
    print("PASS ✅" if passed else "FAIL ❌")
    print()


def test_gac_root_level():
    print("=== GAC Root-Level Pruning ===")
    x = Variable("X", [1, 2])
    y = Variable("Y", [1])
    c = Constraint("C", [x, y])
    c.add_satisfying_tuples([(1, 1)])

    csp = CSP("GAC_Root", [x, y])
    csp.add_constraint(c)

    status, pruned = prop_GAC(csp)
    expected = [1]
    passed = x.cur_domain() == expected and y.cur_domain() == [1]

    print("X domain:", x.cur_domain())
    print("Y domain:", y.cur_domain())
    print("PASS ✅" if passed else "FAIL ❌")
    print()


def test_fc_deadend():
    print("=== FC Dead-End Detection (Domain Wipeout) ===")
    v1 = Variable("V1", [1])
    v2 = Variable("V2", [1])
    c = Constraint("C1", [v1, v2])
    c.add_satisfying_tuples([(1, 2)])  # No tuple is possible
    csp = CSP("FC_Deadend", [v1, v2])
    csp.add_constraint(c)
    v1.assign(1)
    status, pruned = prop_FC(csp, newVar=v1)
    print("Status:", status)
    print("V2 domain:", v2.cur_domain())
    print("PASS ✅" if status is False and v2.cur_domain() == [] else "FAIL ❌")
    print()


def test_gac_deadend():
    print("=== GAC Dead-End Detection (Domain Wipeout) ===")
    x = Variable("X", [1])
    y = Variable("Y", [1])
    c = Constraint("C", [x, y])
    c.add_satisfying_tuples([(2, 2)])  # No tuple is possible
    csp = CSP("GAC_Deadend", [x, y])
    csp.add_constraint(c)
    x.assign(1)
    status, pruned = prop_GAC(csp, newVar=x)
    print("Status:", status)
    print("Y domain:", y.cur_domain())
    print("PASS ✅" if status is False and y.cur_domain() == [] else "FAIL ❌")
    print()


def test_fc_unary_constraint():
    print("=== FC with Unary Constraint ===")
    v = Variable("V", [1, 2, 3])
    c = Constraint("C_unary", [v])
    c.add_satisfying_tuples([(2,), (3,)])
    csp = CSP("FC_Unary", [v])
    csp.add_constraint(c)
    status, pruned = prop_FC(csp)
    print("V domain:", v.cur_domain())
    print("PASS ✅" if v.cur_domain() == [2, 3] else "FAIL ❌")
    print()


def test_gac_ternary_all_diff():
    print("=== GAC with Ternary All-Different Constraint ===")
    x = Variable("X", [1, 2, 3])
    y = Variable("Y", [1, 2, 3])
    z = Variable("Z", [1, 2, 3])
    c = Constraint("AllDiff", [x, y, z])
    import itertools
    c.add_satisfying_tuples([t for t in itertools.permutations([1, 2, 3], 3)])
    csp = CSP("GAC_AllDiff", [x, y, z])
    csp.add_constraint(c)
    x.assign(1)
    status, pruned = prop_GAC(csp, newVar=x)
    print("Y domain:", y.cur_domain())
    print("Z domain:", z.cur_domain())
    passed = (sorted(y.cur_domain()) == [2, 3] and sorted(z.cur_domain()) == [2, 3])
    print("PASS ✅" if passed else "FAIL ❌")
    print()


def test_gac_cycle_binary_constraints():
    print("=== GAC with Cycle of Binary Not-Equal Constraints ===")
    x = Variable("X", [1, 2])
    y = Variable("Y", [1, 2])
    z = Variable("Z", [1, 2])
    c1 = Constraint("X!=Y", [x, y])
    c2 = Constraint("Y!=Z", [y, z])
    c3 = Constraint("Z!=X", [z, x])
    c1.add_satisfying_tuples([(1, 2), (2, 1)])
    c2.add_satisfying_tuples([(1, 2), (2, 1)])
    c3.add_satisfying_tuples([(1, 2), (2, 1)])
    csp = CSP("GAC_Cycle", [x, y, z])
    csp.add_constraint(c1)
    csp.add_constraint(c2)
    csp.add_constraint(c3)
    status, pruned = prop_GAC(csp)
    print("X domain:", x.cur_domain())
    print("Y domain:", y.cur_domain())
    print("Z domain:", z.cur_domain())
    passed = (sorted(x.cur_domain()) == [1, 2] and sorted(y.cur_domain()) == [1, 2] and sorted(z.cur_domain()) == [1, 2])
    print("PASS ✅" if passed else "FAIL ❌")
    print()



def test_gac_large_arity_all_diff():
    print("=== GAC Large Arity All-Different with Partial Assignment ===")
    n = 5
    vars = [Variable(f"V{i}", list(range(1, n+1))) for i in range(n)]
    c = Constraint("AllDiff", vars)
    import itertools
    c.add_satisfying_tuples([t for t in itertools.permutations(range(1, n+1), n)])
    csp = CSP("GAC_LargeAllDiff", vars)
    csp.add_constraint(c)
    vars[0].assign(1)
    vars[1].assign(2)
    status, pruned = prop_GAC(csp, newVar=vars[1])
    expected_domains = [[1], [2], [3, 4, 5], [3, 4, 5], [3, 4, 5]]
    actual_domains = [v.cur_domain() for v in vars]
    passed = actual_domains == expected_domains
    print("Domains:", actual_domains)
    print("PASS ✅" if passed else "FAIL ❌")
    print()


def test_fc_chain_propagation():
    print("=== FC Chain Propagation Wipeout ===")
    v1 = Variable("V1", [1, 2])
    v2 = Variable("V2", [1, 2])
    v3 = Variable("V3", [1, 2])
    c1 = Constraint("C1", [v1, v2])
    c2 = Constraint("C2", [v2, v3])
    c1.add_satisfying_tuples([(1, 1), (2, 2)])
    c2.add_satisfying_tuples([(1, 2)])
    csp = CSP("FC_Chain", [v1, v2, v3])
    csp.add_constraint(c1)
    csp.add_constraint(c2)
    v1.assign(2)
    status, pruned = prop_FC(csp, newVar=v1)
    print("Status:", status)
    print("V2 domain:", v2.cur_domain())
    print("V3 domain:", v3.cur_domain())
    # For standard FC, no chain propagation, so expect status True and no wipeout
    print("PASS ✅" if status is True and v3.cur_domain() == [1, 2] else "FAIL ❌")
    print()


def test_gac_overlapping_constraints():
    print("=== GAC Overlapping Constraints Interaction ===")
    x = Variable("X", [1, 2, 3])
    y = Variable("Y", [1, 2, 3])
    z = Variable("Z", [1, 2, 3])
    c1 = Constraint("C1", [x, y])
    c2 = Constraint("C2", [y, z])
    c3 = Constraint("C3", [x, z])
    c1.add_satisfying_tuples([(1, 2), (2, 3), (3, 1)])
    c2.add_satisfying_tuples([(2, 1), (3, 2), (1, 3)])
    c3.add_satisfying_tuples([(1, 3), (2, 1), (3, 2)])
    csp = CSP("GAC_Overlap", [x, y, z])
    csp.add_constraint(c1)
    csp.add_constraint(c2)
    csp.add_constraint(c3)
    status, pruned = prop_GAC(csp)
    print("X domain:", x.cur_domain())
    print("Y domain:", y.cur_domain())
    print("Z domain:", z.cur_domain())
    # All domains should be pruned to [1,2,3] but only values that are in at least one tuple in all constraints
    passed = (x.cur_domain() == [1,2,3] and y.cur_domain() == [1,2,3] and z.cur_domain() == [1,2,3])
    print("PASS ✅" if passed else "FAIL ❌")
    print()


def test_fc_empty_domain_start():
    print("=== FC Variable Starts with Empty Domain ===")
    v = Variable("V", [])
    c = Constraint("C", [v])
    c.add_satisfying_tuples([])
    csp = CSP("FC_EmptyDomain", [v])
    csp.add_constraint(c)
    status, pruned = prop_FC(csp)
    print("Status:", status)
    print("V domain:", v.cur_domain())
    # FC should return False if any variable starts with empty domain
    print("PASS ✅" if status is False and v.cur_domain() == [] else "FAIL ❌")
    print()


def test_gac_always_unsat():
    print("=== GAC Always Unsatisfiable Constraint ===")
    x = Variable("X", [1, 2])
    y = Variable("Y", [1, 2])
    c = Constraint("C_unsat", [x, y])
    c.add_satisfying_tuples([])
    csp = CSP("GAC_Unsat", [x, y])
    csp.add_constraint(c)
    status, pruned = prop_GAC(csp)
    print("Status:", status)
    print("X domain:", x.cur_domain())
    print("Y domain:", y.cur_domain())
    # GAC should return False if any domain is empty
    print("PASS ✅" if status is False and (x.cur_domain() == [] or y.cur_domain() == []) else "FAIL ❌")
    print()


def test_gac_always_tautology():
    print("=== GAC Always Satisfied (Tautology) Constraint ===")
    x = Variable("X", [1, 2, 3])
    y = Variable("Y", [1, 2, 3])
    c = Constraint("C_taut", [x, y])
    c.add_satisfying_tuples([(a, b) for a in [1,2,3] for b in [1,2,3]])
    csp = CSP("GAC_Tautology", [x, y])
    csp.add_constraint(c)
    status, pruned = prop_GAC(csp)
    print("Status:", status)
    print("X domain:", x.cur_domain())
    print("Y domain:", y.cur_domain())
    print("PASS ✅" if status and x.cur_domain() == [1,2,3] and y.cur_domain() == [1,2,3] else "FAIL ❌")
    print()


def test_gac_var_not_in_constraint():
    print("=== GAC Variable Not in Any Constraint ===")
    x = Variable("X", [1, 2])
    y = Variable("Y", [1, 2])
    c = Constraint("C", [y])
    c.add_satisfying_tuples([(1,), (2,)])
    csp = CSP("GAC_VarNoCons", [x, y])
    csp.add_constraint(c)
    status, pruned = prop_GAC(csp)
    print("Status:", status)
    print("X domain:", x.cur_domain())
    print("Y domain:", y.cur_domain())
    print("PASS ✅" if x.cur_domain() == [1,2] and y.cur_domain() == [1,2] else "FAIL ❌")
    print()


def test_fc_single_tuple_constraint():
    print("=== FC Constraint with Only One Satisfying Tuple ===")
    v1 = Variable("V1", [1, 2, 3])
    v2 = Variable("V2", [1, 2, 3])
    c = Constraint("C_single", [v1, v2])
    c.add_satisfying_tuples([(2, 3)])
    csp = CSP("FC_SingleTuple", [v1, v2])
    csp.add_constraint(c)
    status, pruned = prop_FC(csp)
    print("V1 domain:", v1.cur_domain())
    print("V2 domain:", v2.cur_domain())
    # FC should not prune when both variables are unassigned
    print("PASS ✅" if v1.cur_domain() == [1, 2, 3] and v2.cur_domain() == [1, 2, 3] and status is True else "FAIL ❌")
    print()


def test_gac_prune_to_one_value():
    print("=== GAC Constraint Prunes All But One Value ===")
    x = Variable("X", [1, 2, 3])
    y = Variable("Y", [1, 2, 3])
    c = Constraint("C_prune1", [x, y])
    c.add_satisfying_tuples([(2, 3)])
    csp = CSP("GAC_Prune1", [x, y])
    csp.add_constraint(c)
    status, pruned = prop_GAC(csp)
    print("X domain:", x.cur_domain())
    print("Y domain:", y.cur_domain())
    print("PASS ✅" if x.cur_domain() == [2] and y.cur_domain() == [3] else "FAIL ❌")
    print()


def test_gac_large_domain_few_tuples():
    print("=== GAC Large Domain, Few Satisfying Tuples ===")
    x = Variable("X", list(range(1, 11)))
    y = Variable("Y", list(range(1, 11)))
    c = Constraint("C_few", [x, y])
    c.add_satisfying_tuples([(1, 10), (10, 1)])
    csp = CSP("GAC_FewTuples", [x, y])
    csp.add_constraint(c)
    status, pruned = prop_GAC(csp)
    print("X domain:", x.cur_domain())
    print("Y domain:", y.cur_domain())
    print("PASS ✅" if sorted(x.cur_domain()) == [1, 10] and sorted(y.cur_domain()) == [1, 10] else "FAIL ❌")
    print()


def test_stress_large_domain():
    print("=== Stress Test: Large Domain with Binary Constraint ===")
    size = 50
    x = Variable("X", list(range(size)))
    y = Variable("Y", list(range(size)))
    c = Constraint("X!=Y", [x, y])
    c.add_satisfying_tuples([(a, b) for a in range(size) for b in range(size) if a != b])
    csp = CSP("Stress_LargeDomain", [x, y])
    csp.add_constraint(c)
    status, pruned = prop_GAC(csp)
    # Should not prune anything, all values have support
    print("X domain size:", len(x.cur_domain()))
    print("Y domain size:", len(y.cur_domain()))
    print("PASS ✅" if len(x.cur_domain()) == size and len(y.cur_domain()) == size else "FAIL ❌")
    print()


def test_multiple_overlapping_constraints():
    print("=== Multiple Overlapping Constraints ===")
    x = Variable("X", [1, 2, 3])
    y = Variable("Y", [1, 2, 3])
    z = Variable("Z", [2, 3, 4])
    c1 = Constraint("X+Y=Z", [x, y, z])
    c1.add_satisfying_tuples([(a, b, c) for a in [1,2,3] for b in [1,2,3] for c in [2,3,4] if a+b==c])
    c2 = Constraint("X!=Y", [x, y])
    c2.add_satisfying_tuples([(a, b) for a in [1,2,3] for b in [1,2,3] if a != b])
    c3 = Constraint("Y!=Z", [y, z])
    c3.add_satisfying_tuples([(b, c) for b in [1,2,3] for c in [2,3,4] if b != c])
    csp = CSP("Overlap_Multi", [x, y, z])
    csp.add_constraint(c1)
    csp.add_constraint(c2)
    csp.add_constraint(c3)
    status, pruned = prop_GAC(csp)
    # Just check that domains are not empty and all values are consistent
    print("X domain:", x.cur_domain())
    print("Y domain:", y.cur_domain())
    print("Z domain:", z.cur_domain())
    print("PASS ✅" if all(len(v.cur_domain()) > 0 for v in [x, y, z]) else "FAIL ❌")
    print()


def test_redundant_constraints():
    print("=== Redundant Constraints (Identical All-Diff) ===")
    x = Variable("X", [1, 2, 3])
    y = Variable("Y", [1, 2, 3])
    z = Variable("Z", [1, 2, 3])
    import itertools
    tuples = [t for t in itertools.permutations([1,2,3], 3)]
    c1 = Constraint("AllDiff1", [x, y, z])
    c2 = Constraint("AllDiff2", [x, y, z])
    c1.add_satisfying_tuples(tuples)
    c2.add_satisfying_tuples(tuples)
    csp = CSP("Redundant_AllDiff", [x, y, z])
    csp.add_constraint(c1)
    csp.add_constraint(c2)
    status, pruned = prop_GAC(csp)
    print("X domain:", x.cur_domain())
    print("Y domain:", y.cur_domain())
    print("Z domain:", z.cur_domain())
    print("PASS ✅" if sorted(x.cur_domain()) == [1,2,3] else "FAIL ❌")
    print()


def test_all_vars_assigned():
    print("=== All Variables Assigned ===")
    x = Variable("X", [1, 2])
    y = Variable("Y", [1, 2])
    c = Constraint("X!=Y", [x, y])
    c.add_satisfying_tuples([(1, 2), (2, 1)])
    csp = CSP("AllAssigned", [x, y])
    csp.add_constraint(c)
    x.assign(1)
    y.assign(2)
    status, pruned = prop_GAC(csp)
    print("Status:", status)
    print("PASS ✅" if status else "FAIL ❌")
    print()


def test_no_constraints():
    print("=== No Constraints ===")
    x = Variable("X", [1, 2, 3])
    y = Variable("Y", [1, 2, 3])
    csp = CSP("NoCons", [x, y])
    status, pruned = prop_GAC(csp)
    print("X domain:", x.cur_domain())
    print("Y domain:", y.cur_domain())
    print("PASS ✅" if x.cur_domain() == [1,2,3] and y.cur_domain() == [1,2,3] else "FAIL ❌")
    print()


def test_empty_scope_constraint():
    print("=== Constraint with No Variables (Empty Scope) ===")
    csp = CSP("EmptyScope", [])
    c = Constraint("Empty", [])
    c.add_satisfying_tuples([()])
    csp.add_constraint(c)
    status, pruned = prop_GAC(csp)
    print("Status:", status)
    print("PASS ✅" if status else "FAIL ❌")
    print()


def test_complex_unary_constraint():
    print("=== Complex Unary Constraint (Even Numbers Only) ===")
    x = Variable("X", [1, 2, 3, 4, 5, 6])
    c = Constraint("Even", [x])
    c.add_satisfying_tuples([(v,) for v in [2,4,6]])
    csp = CSP("UnaryEven", [x])
    csp.add_constraint(c)
    status, pruned = prop_FC(csp)
    print("X domain:", x.cur_domain())
    print("PASS ✅" if x.cur_domain() == [2,4,6] else "FAIL ❌")
    print()


def test_custom_nary_sum_constraint():
    print("=== Custom N-ary Sum Constraint ===")
    x = Variable("X", [1, 2, 3])
    y = Variable("Y", [1, 2, 3])
    z = Variable("Z", [1, 2, 3])
    c = Constraint("Sum=6", [x, y, z])
    c.add_satisfying_tuples([(a, b, c_) for a in [1,2,3] for b in [1,2,3] for c_ in [1,2,3] if a+b+c_==6])
    csp = CSP("Sum6", [x, y, z])
    csp.add_constraint(c)
    status, pruned = prop_GAC(csp)
    print("X domain:", x.cur_domain())
    print("Y domain:", y.cur_domain())
    print("Z domain:", z.cur_domain())
    print("PASS ✅" if sorted(x.cur_domain()) == [1,2,3] and sorted(y.cur_domain()) == [1,2,3] and sorted(z.cur_domain()) == [1,2,3] else "FAIL ❌")
    print()


def test_randomized_csp():
    print("=== Randomized CSP (Crash/No Hang Test) ===")
    import random
    n = 5
    vars = [Variable(f"V{i}", list(range(1, 6))) for i in range(n)]
    csp = CSP("RandomCSP", vars)
    for _ in range(5):
        scope = random.sample(vars, random.randint(2, n))
        tuples = [tuple(random.sample(range(1, 6), len(scope))) for _ in range(5)]
        c = Constraint(f"Rand{_}", scope)
        c.add_satisfying_tuples(tuples)
        csp.add_constraint(c)
    try:
        status, pruned = prop_GAC(csp)
        print("PASS ✅ (No crash)")
    except Exception as e:
        print("FAIL ❌ (Exception)", e)
    print()


if __name__ == "__main__":
    test_empty_domain_after_fc()
    test_gac_multiple_constraints()
    test_gac_root_level()
    test_fc_deadend()
    test_gac_deadend()
    test_fc_unary_constraint()
    test_gac_ternary_all_diff()
    test_gac_cycle_binary_constraints()
    test_gac_large_arity_all_diff()
    test_fc_chain_propagation()
    test_gac_overlapping_constraints()
    test_fc_empty_domain_start()
    test_gac_always_unsat()
    test_gac_always_tautology()
    test_gac_var_not_in_constraint()
    test_fc_single_tuple_constraint()
    test_gac_prune_to_one_value()
    test_gac_large_domain_few_tuples()
    test_stress_large_domain()
    test_multiple_overlapping_constraints()
    test_redundant_constraints()
    test_all_vars_assigned()
    test_no_constraints()
    test_empty_scope_constraint()
    test_complex_unary_constraint()
    test_custom_nary_sum_constraint()
    test_randomized_csp()
