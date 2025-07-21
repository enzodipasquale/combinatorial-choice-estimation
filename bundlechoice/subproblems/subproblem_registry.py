from .registry.linear_knapsack import LinearKnapsackSubproblem
from .registry.greedy import GreedySubproblem
from .registry.quad_supermod_network import QuadSupermodularSubproblem
from bundlechoice.subproblems.registry.plain_single_item import PlainSingleItemSubproblem

SUBPROBLEM_REGISTRY = {
    "LinearKnapsack": LinearKnapsackSubproblem,
    "Greedy": GreedySubproblem,
    "QuadSupermodularNetwork": QuadSupermodularSubproblem,
    'PlainSingleItem': PlainSingleItemSubproblem,
} 