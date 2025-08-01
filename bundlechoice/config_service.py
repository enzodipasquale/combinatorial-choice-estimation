"""
Configuration service for modular bundle choice estimation.

This module provides a service-oriented approach to configuration management
that supports dependency injection and dynamic configuration resolution.
"""
from typing import Dict, Any, Optional, Type, TypeVar, Callable
from dataclasses import dataclass
from .config import BundleChoiceConfig, ConfigRegistry

T = TypeVar('T')

@dataclass
class ConfigDependency:
    """Represents a configuration dependency with optional validation."""
    name: str
    config_type: Type[T]
    required: bool = True
    validator: Optional[Callable[[T], bool]] = None
    default: Optional[T] = None

class ConfigurationService:
    """
    Service for managing configuration dependencies and injection.
    
    This class provides a clean interface for registering and resolving
    configuration dependencies, supporting both static and dynamic configuration.
    """
    
    def __init__(self):
        self._dependencies: Dict[str, ConfigDependency] = {}
        self._resolvers: Dict[str, Callable] = {}
        self._cache: Dict[str, Any] = {}
    
    def register_dependency(self, dependency: ConfigDependency):
        """Register a configuration dependency."""
        self._dependencies[dependency.name] = dependency
    
    def register_resolver(self, name: str, resolver: Callable):
        """Register a resolver function for dynamic configuration."""
        self._resolvers[name] = resolver
    
    def get_config(self, name: str) -> Optional[Any]:
        """Get a configuration by name, using cache if available."""
        if name in self._cache:
            return self._cache[name]
        
        if name in self._resolvers:
            config = self._resolvers[name]()
            self._cache[name] = config
            return config
        
        return None
    
    def get_typed_config(self, name: str, config_type: Type[T]) -> Optional[T]:
        """Get a typed configuration with validation."""
        config = self.get_config(name)
        if config is not None and isinstance(config, config_type):
            return config
        return None
    
    def inject_configs(self, target: Any, config_names: list[str]):
        """Inject configurations into a target object."""
        for name in config_names:
            config = self.get_config(name)
            if config is not None:
                setattr(target, f"{name}_cfg", config)
    
    def validate_dependencies(self, config: BundleChoiceConfig) -> Dict[str, bool]:
        """Validate all registered dependencies against a configuration."""
        results = {}
        for name, dependency in self._dependencies.items():
            config_value = getattr(config, name, None)
            if dependency.required and config_value is None:
                results[name] = False
            elif config_value is not None and dependency.validator:
                results[name] = dependency.validator(config_value)
            else:
                results[name] = True
        return results

class ConfigurableMixin:
    """
    Mixin that provides configuration injection capabilities.
    
    Classes using this mixin can easily receive and validate
    configuration dependencies.
    """
    
    def __init__(self):
        self._config_service: Optional[ConfigurationService] = None
        self._injected_configs: Dict[str, Any] = {}
    
    def set_config_service(self, service: ConfigurationService):
        """Set the configuration service for this object."""
        self._config_service = service
    
    def inject_config(self, name: str, config: Any):
        """Inject a configuration directly."""
        self._injected_configs[name] = config
        setattr(self, f"{name}_cfg", config)
    
    def get_config(self, name: str, default=None):
        """Get a configuration by name."""
        if name in self._injected_configs:
            return self._injected_configs[name]
        if self._config_service:
            return self._config_service.get_config(name)
        return default
    
    def get_typed_config(self, name: str, config_type: Type[T]) -> Optional[T]:
        """Get a typed configuration."""
        config = self.get_config(name)
        if config is not None and isinstance(config, config_type):
            return config
        return None
    
    def validate_required_configs(self, required_configs: list[str]) -> bool:
        """Validate that all required configurations are present."""
        for config_name in required_configs:
            if self.get_config(config_name) is None:
                return False
        return True

# Global configuration service instance
config_service = ConfigurationService()

# Register common dependencies
config_service.register_dependency(ConfigDependency(
    name="dimensions",
    config_type=Any,  # Will be typed properly in usage
    required=True
))

config_service.register_dependency(ConfigDependency(
    name="subproblem", 
    config_type=Any,
    required=True
))

config_service.register_dependency(ConfigDependency(
    name="row_generation",
    config_type=Any,
    required=False
))

config_service.register_dependency(ConfigDependency(
    name="ellipsoid",
    config_type=Any, 
    required=False
)) 