import importlib

REGISTRY = {
    'LinearKnapsackGRB': ('.registry.quadratic_obj.linear_knapsack', 'LinearKnapsackGRBSolver'),
    'Greedy': ('.registry.greedy', 'GreedySolver'),
    'QuadraticKnapsackGRB': ('.registry.quadratic_obj.quadratic_knapsack', 'QuadraticKnapsackGRBSolver'),
    'QuadraticSupermodularMinCut': ('.registry.quadratic_obj.quadratic_supermodular', 'QuadraticSupermodularMinCutSolver'),
    'QuadraticSupermodularLovasz': ('.registry.quadratic_obj.quadratic_supermodular', 'QuadraticSupermodularLovaszSolver'),
    'UnitDemand': ('.registry.quadratic_obj.unit_demand', 'UnitDemandSolver'),
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
        module = importlib.import_module(module_path, package='combest.subproblems')
        _cache[name] = getattr(module, class_name)
        return _cache[name]

    def keys(self):
        return REGISTRY.keys()

    def __contains__(self, name):
        return name in REGISTRY

SOLVER_REGISTRY = LazyRegistry()
