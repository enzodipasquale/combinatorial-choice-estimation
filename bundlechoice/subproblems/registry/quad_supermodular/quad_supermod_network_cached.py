"""
Quadratic Supermodular Network Subproblem Solver (Cached)
----------------------------------------------------------
Optimized version that builds the graph once and only updates edge weights on subsequent solves.
"""
import numpy as np
import networkx as nx
from typing import Any, Optional
from .quad_supermod_network import MinCutSubmodularSolver
from .quadratic_supermodular_base import QuadraticSupermodular


class QuadraticSOptNetworkCached(QuadraticSupermodular):
    """
    Cached version of quadratic supermodular minimization that reuses graph structure.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._solvers = None
    
    def initialize(self):
        """Initialize cached solvers for each local agent."""
        super().initialize()
        
        # Build solvers once with initial parameters
        P_i_j_j = self.build_quadratic_matrix(np.zeros(self.num_features))
        agent_data = self.local_data.get("agent_data") or {}
        constraint_mask = agent_data.get("constraint_mask") if self.has_constraint_mask else None
        
        self._solvers = []
        for i in range(self.num_local_agents):
            mask_i = constraint_mask[i] if constraint_mask is not None else None
            solver = MinCutSubmodularSolverCached(-P_i_j_j[i], mask_i)
            self._solvers.append(solver)
    
    def solve(self, theta: np.ndarray, pb: Optional[Any] = None) -> np.ndarray:
        if self._solvers is None:
            self.initialize()
        
        P_i_j_j = self.build_quadratic_matrix(theta)
        optimal_bundles = np.zeros((self.num_local_agents, self.num_items), dtype=bool)
        
        for i in range(self.num_local_agents):
            self._solvers[i].update_weights(-P_i_j_j[i])
            optimal_bundle = self._solvers[i].solve_QSM()
            optimal_bundles[i] = optimal_bundle
        
        return optimal_bundles


class MinCutSubmodularSolverCached(MinCutSubmodularSolver):
    """
    Cached version that builds the graph once and updates only edge weights.
    """
    def __init__(self, b_j_j: np.ndarray, constraint_mask: Optional[np.ndarray] = None):
        super().__init__(b_j_j, constraint_mask)
        
        # Cache triangular matrix for posiform computation
        self._triu_mask = np.triu(np.ones((self.num_items, self.num_items)), k=1)
        
        # Build fixed graph structure with ALL possible edges
        # This avoids add/remove operations - we just update capacities
        self.G = self._build_fixed_graph()
        
        # Store direct edge references for efficient updates
        self.edge_structure = self._store_edge_references()
        
        # Track which source/sink edges currently exist (avoid expensive has_edge calls)
        self._existing_source_edges = set()
        self._existing_sink_edges = set()
        
        # Initialize with current weights (this will set initial cached values)
        self.update_weights(b_j_j)
    
    def _build_fixed_graph(self):
        """Build graph with all pair edges (these are always needed), but not source/sink edges initially."""
        G = nx.DiGraph()
        G.add_node('s')
        G.add_node('t')
        G.add_nodes_from(self.choice_set)
        
        # Add ALL pair edges (these always exist for the quadratic terms)
        # Source/sink edges will be added/removed as needed during update_weights
        for i in self.choice_set:
            for j in self.choice_set:
                if j > i:
                    G.add_edge(i, j, capacity=0)
        
        # Source->sink edge
        G.add_edge('s', 't', capacity=0)
        
        return G
    
    def _store_edge_references(self):
        """Store direct edge references for efficient weight updates."""
        # Track which source/sink edges currently exist
        pair_edges = {}    # Maps (i, j) to edge data dict
        
        for i in self.choice_set:
            for j in self.choice_set:
                if j > i:
                    pair_edges[(i, j)] = self.G[i][j]
        
        return {
            'pair_edges': pair_edges,
            'source_sink_edge': self.G['s']['t']
        }
    
    def _round_capacities(self, a_j, a_j_j):
        """Robust rounding with overflow checking (optimized version)."""
        scale = 1e12
        a_j_scaled = a_j * scale
        a_j_j_scaled = a_j_j * scale
        
        # Find max absolute value more efficiently
        max_val = max(
            np.abs(a_j_scaled).max() if a_j.size > 0 else 0,
            np.abs(a_j_j_scaled).max() if a_j_j_scaled.size > 0 else 0
        )
        
        if max_val > np.iinfo(np.int64).max:
            safe_scale = np.iinfo(np.int64).max / (max_val + 1)
            scale = scale * safe_scale
            a_j_int = np.round(a_j * scale).astype(np.int64)
            a_j_j_int = np.round(a_j_j * scale).astype(np.int64)
        else:
            a_j_int = np.round(a_j_scaled).astype(np.int64)
            a_j_j_int = np.round(a_j_j_scaled).astype(np.int64)
        
        return a_j_int, a_j_j_int, scale
    
    def update_weights(self, b_j_j: np.ndarray):
        """Update edge weights - efficiently add/remove source/sink edges as needed."""
        self.b_j_j = b_j_j
        
        # Update cached posiform (using cached triangular matrix)
        self._cached_a_j_j = -self.b_j_j * self._triu_mask
        self._cached_a_j = np.diag(self.b_j_j) - self._cached_a_j_j.sum(axis=1)
        
        a_j_int, a_j_j_int, _ = self._round_capacities(self._cached_a_j, self._cached_a_j_j)
        
        # Update node edges: add/remove source or sink edges based on sign of a_j
        # This keeps the graph sparse (no unused edges) for faster min-cut
        for i in self.choice_set:
            if self._cached_a_j[i] >= 0:
                # Need sink edge, remove source if exists
                if i in self._existing_source_edges:
                    self.G.remove_edge('s', i)
                    self._existing_source_edges.remove(i)
                if i not in self._existing_sink_edges:
                    self.G.add_edge(i, 't', capacity=a_j_int[i])
                    self._existing_sink_edges.add(i)
                else:
                    self.G[i]['t']['capacity'] = a_j_int[i]
            else:
                # Need source edge, remove sink if exists
                if i in self._existing_sink_edges:
                    self.G.remove_edge(i, 't')
                    self._existing_sink_edges.remove(i)
                if i not in self._existing_source_edges:
                    self.G.add_edge('s', i, capacity=-a_j_int[i])
                    self._existing_source_edges.add(i)
                else:
                    self.G['s'][i]['capacity'] = -a_j_int[i]
        
        # Update pair-to-pair edges using cached references
        pair_edges = self.edge_structure['pair_edges']
        for (i, j), edge_data in pair_edges.items():
            edge_data['capacity'] = a_j_j_int[i, j]
        
        # Source->sink edge always 0
        self.edge_structure['source_sink_edge']['capacity'] = 0
    
    @staticmethod
    def solve_mincut(G):
        """
        Solve the min-cut problem (same as base class for consistency).
        """
        # Use push_relabel instead of edmonds_karp for better performance on dense graphs
        cut_value, partition = nx.minimum_cut(G, 's', 't', flow_func=nx.algorithms.flow.preflow_push)
        S, T = partition
        S = list(S - {'s'})
        return S, cut_value
    
    def solve_QSM(self):
        """Solve using the cached graph with updated weights."""
        S, cut_value = self.solve_mincut(self.G)
        optimal_bundle = np.zeros(self.num_items, dtype=bool)
        optimal_bundle[S] = True
        return optimal_bundle
