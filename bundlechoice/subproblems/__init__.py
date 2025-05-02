def get_subproblem(name):
    if name == "QuadKnap":
        from .quadratic_knapsack import init_QKP, solve_QKP
        return init_QKP, solve_QKP

    elif name == "UnconstrSupermod":
        from .quadratic_supermod import solve_QS
        return None, solve_USM
        
    else:
        raise ValueError(f"Unknown subproblem: {name}")
