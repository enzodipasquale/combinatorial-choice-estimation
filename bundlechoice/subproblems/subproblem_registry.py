import importlib

REGISTRY = {
    'LinearKnapsack': ('.registry.linear_knapsack', 'LinearKnapsackSubproblem'),
    'Greedy': ('.registry.greedy', 'GreedySubproblem'),
    'QuadKnapsack': ('.registry.quadratic_obj.quadratic_knapsack', 'QuadraticKnapsackSubproblem'),
    'QuadSupermodularNetwork': ('.registry.quadratic_obj.quadratic_supermodular', 'QuadraticSupermodularMinCut'),
    'QuadSupermodularLovasz': ('.registry.quadratic_obj.quadratic_supermodular', 'QuadraticSupermodularLovasz'),
    'PlainSingleItem': ('.registry.quadratic_obj.plain_single_item', 'PlainSingleItemSubproblem'),
    'BruteForce': ('.registry.brute_force', 'BruteForceSubproblem'),
}
_cache = {}

class LazyRegistry:

    def get(self, name):
        if name in _cache:
            return _cache[name]
        if name not in REGISTRY:
            return None
        module_path, class_name = REGISTRY[name]
        module = importlib.import_module(module_path, package='bundlechoice.subproblems')
        _cache[name] = getattr(module, class_name)
        return _cache[name]

    def keys(self):
        return REGISTRY.keys()

    def __contains__(self, name):
        return name in REGISTRY

SUBPROBLEM_REGISTRY = LazyRegistry()
