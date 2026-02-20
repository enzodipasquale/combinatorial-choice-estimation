import importlib

REGISTRY = {
    'LinearKnapsackGRB': ('.registry.quadratic_obj.linear_knapsack', 'LinearKnapsackGRBSolver'),
    'Greedy': ('.registry.greedy', 'GreedySolver'),
    'QuadraticKnapsackGRB': ('.registry.quadratic_obj.quadratic_knapsack', 'QuadraticKnapsackGRBSolver'),
    'QuadraticSupermodularMinCut': ('.registry.quadratic_obj.quadratic_supermodular', 'QuadraticSupermodularMinCutSolver'),
    'QuadraticSupermodularLovasz': ('.registry.quadratic_obj.quadratic_supermodular', 'QuadraticSupermodularLovaszSolver'),
    'PlainSingleItem': ('.registry.quadratic_obj.plain_single_item', 'PlainSingleItemSolver'),
    'BruteForce': ('.registry.brute_force', 'BruteForceSolver'),
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
