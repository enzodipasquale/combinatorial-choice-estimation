from .registry.linear_knapsack import LinearKnapsackSubproblem
from .registry.greedy import GreedySubproblem
from .registry.quad_supermod_network import QuadraticSOptNetwork
from .registry.quad_supermod_lovasz import QuadraticSOptLovasz
from bundlechoice.subproblems.registry.plain_single_item import PlainSingleItemSubproblem

SUBPROBLEM_REGISTRY = {
    "LinearKnapsack": LinearKnapsackSubproblem,
    "Greedy": GreedySubproblem,
    "QuadSupermodularNetwork": QuadraticSOptNetwork,
    "QuadSupermodularLovasz": QuadraticSOptLovasz,
    'PlainSingleItem': PlainSingleItemSubproblem,
} 