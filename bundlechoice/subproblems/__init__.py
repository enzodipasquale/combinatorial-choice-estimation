def get_subproblem(name):
    if name == "QuadKnap":
        from .quadratic_knapsack import init_QKP, solve_QKP
        return init_QKP, solve_QKP

    elif name == "QuadSupermod":
        from .quadratic_supermod import solve_QS
        return None, solve_USM
    
    elif name == "Greedy":
        from .greedy import solve_greedy
        return None, solve_greedy
        
    else:
        raise ValueError(f"Unknown subproblem: {name}")
