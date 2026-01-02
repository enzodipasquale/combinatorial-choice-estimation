"""
Lazy subproblem registry - imports implementations only when requested.
"""

from typing import Dict, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseSubproblem


def _lazy_import(module_path: str, class_name: str) -> Type["BaseSubproblem"]:
    """Import a class lazily from a module path."""
    import importlib
    module = importlib.import_module(module_path, package="bundlechoice.subproblems")
    return getattr(module, class_name)


class LazyRegistry:
    """Registry that imports subproblem classes on first access."""
    
    _registry: Dict[str, tuple] = {
        "LinearKnapsack": (".registry.linear_knapsack", "LinearKnapsackSubproblem"),
        "Greedy": (".registry.greedy", "GreedySubproblem"),
        "QuadKnapsack": (".registry.quadratic_knapsack", "QuadraticKnapsackSubproblem"),
        "QuadSupermodularNetwork": (".registry.quad_supermodular", "QuadraticSOptNetwork"),
        "QuadSupermodularLovasz": (".registry.quad_supermodular", "QuadraticSOptLovasz"),
        "PlainSingleItem": (".registry.plain_single_item", "PlainSingleItemSubproblem"),
    }
    _cache: Dict[str, Type["BaseSubproblem"]] = {}
    
    def get(self, name: str) -> Type["BaseSubproblem"] | None:
        """Get subproblem class by name, importing lazily."""
        if name in self._cache:
            return self._cache[name]
        if name not in self._registry:
            return None
        module_path, class_name = self._registry[name]
        cls = _lazy_import(module_path, class_name)
        self._cache[name] = cls
        return cls
    
    def keys(self):
        """Return available subproblem names."""
        return self._registry.keys()
    
    def __contains__(self, name: str) -> bool:
        return name in self._registry


SUBPROBLEM_REGISTRY = LazyRegistry()
