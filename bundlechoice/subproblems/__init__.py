def get_subproblem(name):
    if name == "QuadKnap":
        from .quadratic_knapsack import init_QKP, solve_QKP
        return init_QKP, solve_QKP

    elif name == "UnconstrSupermod":
        from .quadratic_supermod import init_USM, solve_USM
        return init_USM, solve_USM
        
    else:
        raise ValueError(f"Unknown subproblem: {name}")
