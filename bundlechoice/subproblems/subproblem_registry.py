from typing import Dict, Type, TYPE_CHECKING
if TYPE_CHECKING:
    from .base import BaseSubproblem

def _lazy_import(module_path, class_name):
    import importlib
    module = importlib.import_module(module_path, package='bundlechoice.subproblems')
    return getattr(module, class_name)

class LazyRegistry:
    _registry: Dict[str, tuple] = {'LinearKnapsack': ('.registry.linear_knapsack', 'LinearKnapsackSubproblem'), 'Greedy': ('.registry.greedy', 'GreedySubproblem'), 'QuadKnapsack': ('.registry.quadratic_knapsack', 'QuadraticKnapsackSubproblem'), 'QuadSupermodularNetwork': ('.registry.quad_supermodular', 'QuadraticSOptNetwork'), 'QuadSupermodularLovasz': ('.registry.quad_supermodular', 'QuadraticSOptLovasz'), 'PlainSingleItem': ('.registry.plain_single_item', 'PlainSingleItemSubproblem'), 'BruteForce': ('.registry.brute_force', 'BruteForceSubproblem')}
    _cache: Dict[str, Type['BaseSubproblem']] = {}

    def get(self, name):
        if name in self._cache:
            return self._cache[name]
        if name not in self._registry:
            return None
        module_path, class_name = self._registry[name]
        cls = _lazy_import(module_path, class_name)
        self._cache[name] = cls
        return cls

    def keys(self):
        return self._registry.keys()

    def __contains__(self, name):
        return name in self._registry
SUBPROBLEM_REGISTRY = LazyRegistry()