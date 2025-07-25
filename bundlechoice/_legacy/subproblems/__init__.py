def get_subproblem(name):
    if name == "QuadKnap":
        from .quadratic_knapsack import init_QKP, solve_QKP
        return init_QKP, solve_QKP

    elif name == "QuadLovatz":
        from .quad_lovatz import solve_QS
        return None, solve_QS
    
    elif name == "Greedy":
        from .greedy import solve_greedy
        return None, solve_greedy
    
    elif name == "LinearKnap":
        from .linear_knapsack import init_KP, solve_KP
        return init_KP, solve_KP
    elif name == "QuadNetwork":
        from .quad_network import solve_QSNetwork
        return None, solve_QSNetwork
        
    else:
        raise ValueError(f"Unknown subproblem: {name}")
