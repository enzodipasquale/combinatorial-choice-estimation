from .registry.linear_knapsack import LinearKnapsackSubproblem
from .registry.greedy import GreedySubproblem
from .registry.optimized_greedy import OptimizedGreedySubproblem
from .registry.greedy_jit import GreedyJITSubproblem
from .registry.quadratic_knapsack import QuadraticKnapsackSubproblem
from .registry.quad_supermodular import QuadraticSOptNetwork, QuadraticSOptLovasz
from bundlechoice.subproblems.registry.plain_single_item import PlainSingleItemSubproblem

SUBPROBLEM_REGISTRY = {
    "LinearKnapsack": LinearKnapsackSubproblem,
    "Greedy": GreedySubproblem,
    "OptimizedGreedy": OptimizedGreedySubproblem,
    "GreedyJIT": GreedyJITSubproblem,
    "QuadKnapsack": QuadraticKnapsackSubproblem,
    "QuadSupermodularNetwork": QuadraticSOptNetwork,
    "QuadSupermodularLovasz": QuadraticSOptLovasz,
    'PlainSingleItem': PlainSingleItemSubproblem,
} 