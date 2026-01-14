import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from bundlechoice.utils import get_logger
from bundlechoice.config import DimensionsConfig
logger = get_logger(__name__)


class DataManager:
    """
    Handles input data distribution across MPI ranks.
    
    Expected input_data structure:
        {
            'item_data': dict[str, np.ndarray],
            'agent_data': dict[str, np.ndarray],
            'errors': np.ndarray,
            'obs_bundle': np.ndarray,
        }
    """

    def __init__(self, dimensions_cfg: DimensionsConfig, comm_manager) -> None:
        """Initialize DataManager with dimensions config and MPI communicator."""
        self.dimensions_cfg = dimensions_cfg
        self.comm_manager = comm_manager
        self.input_data: Optional[Dict[str, Any]] = None
        self.local_data: Optional[Dict[str, Any]] = None
        self.num_local_agents: Optional[int] = None
        self.data_sources: Dict[str, Dict[str, List[str]]] = {
            "agent_data": {},
            "item_data": {},
        }

    def load(self, input_data: Dict[str, Any], errors_required: bool = True) -> None:
        """
        Load input data (validated, stored on rank 0 only).
        
        Args:
            input_data: Data dictionary with agent_data, item_data, errors, etc.
            errors_required: If False, allow errors to be None (for local generation).
        """
        self._validate_input_data(input_data, errors_required=errors_required)
        if self.comm_manager.is_root():
            self.input_data = input_data
        else:
            self.input_data = None

    def load_and_scatter(self, input_data: Dict[str, Any], errors_required: bool = True) -> Optional[Dict[str, Any]]:
        """
        Load input data and scatter across MPI ranks.
        
        Args:
            input_data: Data dictionary with agent_data, item_data, errors, etc.
            errors_required: If False, allow errors to be None (for local generation
                            via bc.oracles.build_local_modular_error_oracle()).
        """
        self._validate_input_data(input_data, errors_required=errors_required)
        self.load(input_data, errors_required=errors_required)
        self.scatter()
        return self.local_data

    def load_from_directory(
        self,
        path: Union[str, Path],
        generate_errors: bool = True,
        error_seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Load input data from a directory following BundleChoice conventions.
        
        File pattern detection:
            - metadata.json: Dimensions, feature names, and counts
            - matching_*.npy or obs_bundle*.npy: Observed bundles
            - modular_*_i_j_k.npy: Agent modular features
            - modular_*_j_k.npy or modular_*_j.npy: Item modular features
            - quadratic_*_j_j_k.npy: Item quadratic features
            - capacity_*.npy: Agent capacities
            - weight_*.npy: Item weights
        
        Args:
            path: Directory containing .npy files and optional metadata.json
            generate_errors: If True, generate random errors
            error_seed: Random seed for error generation
        
        Returns:
            Loaded input_data dictionary
        """
        path = Path(path)
        
        if self.comm_manager.is_root():
            if not path.exists():
                raise FileNotFoundError(f"Directory not found: {path}")
            
            input_data = self._load_arrays_from_directory(path)
            
            metadata_path = path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                self._update_dimensions_from_metadata(metadata)
            
            if generate_errors:
                if error_seed is not None:
                    np.random.seed(error_seed)
                errors = np.random.normal(
                    0, 1, 
                    (self.dimensions_cfg.num_simulations, self.dimensions_cfg.num_agents, self.dimensions_cfg.num_items)
                )
                input_data["errors"] = errors
        else:
            input_data = None
        
        return input_data
    
    def _find_npy_file(self, npy_files: Dict[str, Path], *patterns: str) -> Optional[Path]:
        """Find first NPY file matching any of the given patterns."""
        for name, path in npy_files.items():
            if all(p in name for p in patterns):
                return path
        return None

    def _load_simple_array(self, csv_files: Dict[str, Path], npy_files: Dict[str, Path],
                           csv_key: str, npy_pattern: str, dtype=None) -> Optional[np.ndarray]:
        """Load array from CSV (first column) or NPY file."""
        if csv_key in csv_files:
            df = pd.read_csv(csv_files[csv_key])
            arr = df.iloc[:, 0].values if df.shape[1] == 1 else df.values
            return arr.astype(dtype) if dtype else arr
        npy_path = self._find_npy_file(npy_files, npy_pattern)
        return np.load(npy_path) if npy_path else None

    def _load_arrays_from_directory(self, path: Path) -> Dict[str, Any]:
        """
        Auto-detect and load arrays from directory. Supports both CSV and NPY.
        
        CSV files use clean names: obs_bundle.csv, modular_agent.csv, quadratic_item.csv
        NPY files use legacy names with suffixes: matching_i_j.npy, modular_*_i_j_k.npy
        
        Feature names are extracted from CSV column headers.
        """
        agent_data = {}
        item_data = {}
        input_data = {}
        
        self.data_sources = {"agent_data": {}, "item_data": {}}
        
        csv_files = {f.stem.lower(): f for f in path.glob("*.csv")}
        npy_files = {f.stem.lower(): f for f in path.glob("*.npy")}
        
        # Observed bundles
        if "obs_bundle" in csv_files:
            input_data["obs_bundle"] = pd.read_csv(csv_files["obs_bundle"]).values.astype(bool)
        else:
            npy_path = self._find_npy_file(npy_files, "matching") or self._find_npy_file(npy_files, "obs_bundle")
            if npy_path:
                input_data["obs_bundle"] = np.load(npy_path)
        
        # Capacity and weights
        capacity = self._load_simple_array(csv_files, npy_files, "capacity", "capacity")
        if capacity is not None:
            agent_data["capacity"] = capacity
        
        weights = self._load_simple_array(csv_files, npy_files, "weight", "weight")
        if weights is not None:
            item_data["weights"] = weights
        
        # Modular agent features
        if "modular_agent" in csv_files:
            arr, feature_names = self._load_agent_features_csv(csv_files["modular_agent"])
            agent_data["modular"] = arr
            self.data_sources["agent_data"]["modular"] = feature_names
        else:
            npy_path = self._find_npy_file(npy_files, "modular", "_i_j_k")
            if npy_path:
                agent_data["modular"] = np.load(npy_path)
        
        # Quadratic item features
        if "quadratic_item" in csv_files:
            arr, feature_names = self._load_item_quadratic_csv(csv_files["quadratic_item"])
            item_data["quadratic"] = arr
            self.data_sources["item_data"]["quadratic"] = feature_names
        else:
            # Match _j_j_k but exclude _i_j_j
            for name, npy_path in npy_files.items():
                if "quadratic" in name and "_j_j_k" in name and "_i_j_j" not in name:
                    item_data["quadratic"] = np.load(npy_path)
                    break
        
        if agent_data:
            input_data["agent_data"] = agent_data
        if item_data:
            input_data["item_data"] = item_data
        
        return input_data
    
    def _load_agent_features_csv(self, csv_path: Path) -> tuple:
        """Load agent modular features from CSV. Returns (array, feature_names)."""
        df = pd.read_csv(csv_path)
        feature_cols = [c for c in df.columns if c not in ("agent_id", "item_id")]
        
        num_agents = df["agent_id"].max() + 1
        num_items = df["item_id"].max() + 1
        num_features = len(feature_cols)
        
        arr = np.zeros((num_agents, num_items, num_features))
        for _, row in df.iterrows():
            i, j = int(row["agent_id"]), int(row["item_id"])
            arr[i, j, :] = [row[c] for c in feature_cols]
        
        return arr, feature_cols
    
    def _load_item_quadratic_csv(self, csv_path: Path) -> tuple:
        """Load item quadratic features from CSV. Returns (array, feature_names)."""
        df = pd.read_csv(csv_path)
        feature_cols = [c for c in df.columns if c not in ("item_i", "item_j")]
        
        num_items = max(df["item_i"].max(), df["item_j"].max()) + 1
        num_features = len(feature_cols)
        
        arr = np.zeros((num_items, num_items, num_features))
        for _, row in df.iterrows():
            i, j = int(row["item_i"]), int(row["item_j"])
            arr[i, j, :] = [row[c] for c in feature_cols]
        
        return arr, feature_cols
    
    def _update_dimensions_from_metadata(self, metadata: Dict[str, Any]) -> None:
        """Update dimensions config from metadata.json."""
        if "num_agents" in metadata:
            self.dimensions_cfg.num_agents = metadata["num_agents"]
        if "num_items" in metadata:
            self.dimensions_cfg.num_items = metadata["num_items"]
        if "num_features" in metadata:
            self.dimensions_cfg.num_features = metadata["num_features"]
        if "feature_names" in metadata:
            self.dimensions_cfg.feature_names = metadata["feature_names"]
        
        if "modular_names" in metadata and "modular" not in self.data_sources.get("agent_data", {}):
            self.data_sources["agent_data"]["modular"] = metadata["modular_names"]
        if "quadratic_names" in metadata and "quadratic" not in self.data_sources.get("item_data", {}):
            self.data_sources["item_data"]["quadratic"] = metadata["quadratic_names"]
        
        if self.dimensions_cfg.num_features is None:
            num_modular = metadata.get("num_modular_features", 0)
            num_quadratic = metadata.get("num_quadratic_features", 0)
            num_items = metadata.get("num_items", 0)
            if num_modular or num_quadratic:
                self.dimensions_cfg.num_features = num_modular + num_items + num_quadratic
    
    def get_feature_names_from_data(self) -> Optional[List[str]]:
        """
        Build feature names from loaded data sources.
        
        Returns names in order: [modular_agent...] + [FE_0...FE_{num_items-1}] + [quadratic_item...]
        """
        if not self.data_sources:
            return None
        
        names = []
        modular_names = self.data_sources.get("agent_data", {}).get("modular", [])
        names.extend(modular_names)
        
        if self.dimensions_cfg.num_items:
            names.extend([f"FE_{j}" for j in range(self.dimensions_cfg.num_items)])
        
        quadratic_names = self.data_sources.get("item_data", {}).get("quadratic", [])
        names.extend(quadratic_names)
        
        return names if names else None

    def scatter(self) -> None:
        """
        Distribute input data across MPI ranks using buffer-based MPI.
        
        Uses buffer operations (5-20x faster than pickle). Each rank receives:
        - Local agent data chunk
        - Local errors array
        - Broadcast item_data (same on all ranks)
        """
        # Root prepares data; non-root will receive via broadcast
        errors = obs_bundles = agent_data = item_data = idx_chunks = counts = None
        has_agent_data = has_obs_bundles = has_item_data = has_errors = False
        
        if self.comm_manager.is_root():
            errors = self._prepare_errors(self.input_data.get("errors"))
            obs_bundles = self.input_data.get("obs_bundle")
            agent_data = self.input_data.get("agent_data") or {}
            item_data = self.input_data.get("item_data")
            
            idx_chunks = np.array_split(np.arange(self.dimensions_cfg.num_simulations * self.dimensions_cfg.num_agents), self.comm_manager.size)
            counts = [len(idx) for idx in idx_chunks]
            
            has_agent_data = len(agent_data) > 0
            has_obs_bundles = obs_bundles is not None
            has_item_data = item_data is not None
            has_errors = errors is not None
            
            total_agents = self.dimensions_cfg.num_simulations * self.dimensions_cfg.num_agents
            sim_info = f" ({self.dimensions_cfg.num_simulations} simuls × {self.dimensions_cfg.num_agents} agents)" if self.dimensions_cfg.num_simulations > 1 else ""
            logger.info("Scattering: %d agents%s → %d ranks", total_agents, sim_info, self.comm_manager.size)
        
        counts, has_agent_data, has_obs_bundles, has_item_data, has_errors = self.comm_manager.broadcast_from_root(
            (counts, has_agent_data, has_obs_bundles, has_item_data, has_errors), root=0
        )
        
        num_local_agents = counts[self.comm_manager.rank]
        flat_counts = [c * self.dimensions_cfg.num_items for c in counts]
        
        if has_errors:
            local_errors_flat = self.comm_manager.scatter_array(
                send_array=errors, counts=flat_counts, root=0, 
                dtype=errors.dtype if self.comm_manager.is_root() else np.float64
            )
            local_errors = local_errors_flat.reshape(num_local_agents, self.dimensions_cfg.num_items)
        else:
            local_errors = None
        
        if has_obs_bundles:
            if self.comm_manager.is_root():
                all_indices = np.concatenate(idx_chunks)
                agent_indices = all_indices % self.dimensions_cfg.num_agents
                indexed_obs_bundles = obs_bundles[agent_indices]
            else:
                indexed_obs_bundles = None
            
            local_obs_bundles_flat = self.comm_manager.scatter_array(
                send_array=indexed_obs_bundles, counts=flat_counts, root=0,
                dtype=indexed_obs_bundles.dtype if self.comm_manager.is_root() else np.bool_
            )
            local_obs_bundles = local_obs_bundles_flat.reshape(num_local_agents, self.dimensions_cfg.num_items)
        else:
            local_obs_bundles = None
        
        if self.comm_manager.is_root():
            global_indices = np.arange(self.dimensions_cfg.num_simulations * self.dimensions_cfg.num_agents, dtype=np.int64)
        else:
            global_indices = None
        local_global_indices = self.comm_manager.scatter_array(
            send_array=global_indices, counts=counts, root=0, dtype=np.int64
        )
        
        if has_agent_data:
            if self.comm_manager.is_root():
                agent_data_expanded = {}
                for k, array in agent_data.items():
                    if self.dimensions_cfg.num_simulations > 1:
                        if array.ndim == 1:
                            agent_data_expanded[k] = np.tile(array, self.dimensions_cfg.num_simulations)
                        else:
                            agent_data_expanded[k] = np.tile(array, (self.dimensions_cfg.num_simulations, 1) + (1,) * (array.ndim - 2))
                    else:
                        agent_data_expanded[k] = array
            else:
                agent_data_expanded = None
            
            local_agent_data = self.comm_manager.scatter_dict(agent_data_expanded, counts=counts, root=0)
        else:
            local_agent_data = None
        
        if has_item_data:
            item_data = self.comm_manager.broadcast_dict(item_data, root=0)
        else:
            item_data = None
        
        self.local_data = {
            "item_data": item_data,
            "agent_data": local_agent_data,
            "errors": local_errors,
            "obs_bundles": local_obs_bundles,
            "global_indices": local_global_indices,
            "agent_counts": counts,  # Per-rank agent counts for scatter operations
        }
        self.num_local_agents = num_local_agents

    def _prepare_errors(self, errors: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if errors is None:
            return None
        if (self.dimensions_cfg.num_simulations == 1 and errors.ndim == 2):
            return errors
        elif errors.ndim == 3:
            return errors.reshape(-1, self.dimensions_cfg.num_items)
        else:
            raise ValueError(f"errors has shape {errors.shape}, while num_simulations is {self.dimensions_cfg.num_simulations} and num_agents is {self.dimensions_cfg.num_agents}")

    def _validate_input_data(self, input_data: Dict[str, Any], errors_required: bool = True) -> None:
        """Validate input_data structure and dimensions (rank 0 only)."""
        if self.comm_manager.is_root():
            from bundlechoice.validation import validate_input_data_comprehensive
            validate_input_data_comprehensive(input_data, self.dimensions_cfg, errors_required=errors_required)

    def verify_feature_count(self) -> None:
        """Verify num_features in config matches data structure."""
        if self.comm_manager.is_root():
            from bundlechoice.validation import validate_feature_count
            validate_feature_count(self.input_data, self.dimensions_cfg.num_features)

    def update_errors(self, errors: np.ndarray) -> None:
        """
        Update errors in local_data by scattering from rank 0.
        
        Args:
            errors: Array of shape (num_agents, num_items) on rank 0, None on other ranks.
        """
        if self.comm_manager.is_root():
            errors_flat = errors.reshape(-1)
            dtype = errors_flat.dtype
            size = self.comm_manager.comm.Get_size()
            idx_chunks = np.array_split(np.arange(self.dimensions_cfg.num_agents), size)
            counts = [len(chunk) * self.dimensions_cfg.num_items for chunk in idx_chunks]
        else:
            errors_flat = None
            dtype = np.float64
            counts = None
        
        local_errors_flat = self.comm_manager.scatter_array(
            errors_flat, counts=counts, root=0, dtype=dtype
        )
        
        if self.num_local_agents == 0:
            self.local_data["errors"] = np.empty((0, self.dimensions_cfg.num_items), dtype=dtype)
        else:
            self.local_data["errors"] = local_errors_flat.reshape(self.num_local_agents, self.dimensions_cfg.num_items)

    def get_data_info(self, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get boolean flags and feature dimensions for data components.
        
        Args:
            data: Data dictionary to check (defaults to local_data).
        
        Returns:
            Dictionary with flags and dimensions.
        """
        if data is None:
            data = self.local_data
        
        if data is None:
            return {
                "has_modular_agent": False,
                "has_modular_item": False,
                "has_quadratic_agent": False,
                "has_quadratic_item": False,
                "has_errors": False,
                "has_constraint_mask": False,
                "num_modular_agent": 0,
                "num_modular_item": 0,
                "num_quadratic_agent": 0,
                "num_quadratic_item": 0,
            }
        
        agent_data = data.get("agent_data") or {}
        item_data = data.get("item_data") or {}
        
        has_modular_agent = "modular" in agent_data
        has_modular_item = "modular" in item_data
        has_quadratic_agent = "quadratic" in agent_data
        has_quadratic_item = "quadratic" in item_data
        
        return {
            "has_modular_agent": has_modular_agent,
            "has_modular_item": has_modular_item,
            "has_quadratic_agent": has_quadratic_agent,
            "has_quadratic_item": has_quadratic_item,
            "has_errors": "errors" in data,
            "has_constraint_mask": "constraint_mask" in agent_data,
            "num_modular_agent": agent_data["modular"].shape[-1] if has_modular_agent else 0,
            "num_modular_item": item_data["modular"].shape[-1] if has_modular_item else 0,
            "num_quadratic_agent": agent_data["quadratic"].shape[-1] if has_quadratic_agent else 0,
            "num_quadratic_item": item_data["quadratic"].shape[-1] if has_quadratic_item else 0,
        }
