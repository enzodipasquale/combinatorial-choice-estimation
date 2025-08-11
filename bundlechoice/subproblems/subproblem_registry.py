from .registry.linear_knapsack import LinearKnapsackSubproblem
from .registry.greedy import GreedySubproblem
from .registry.quadratic_knapsack import QuadraticKnapsackSubproblem
from .registry.quad_supermodular import QuadraticSOptNetwork, QuadraticSOptLovasz
from bundlechoice.subproblems.registry.plain_single_item import PlainSingleItemSubproblem

SUBPROBLEM_REGISTRY = {
    "LinearKnapsack": LinearKnapsackSubproblem,
    "Greedy": GreedySubproblem,
    "QuadKnapsack": QuadraticKnapsackSubproblem,
    "QuadSupermodularNetwork": QuadraticSOptNetwork,
    "QuadSupermodularLovasz": QuadraticSOptLovasz,
    'PlainSingleItem': PlainSingleItemSubproblem,
} 