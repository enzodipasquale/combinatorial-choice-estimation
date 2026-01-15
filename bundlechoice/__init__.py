__version__ = '0.2.0'
__version_info__ = tuple(map(int, __version__.split('.')))
from .core import BundleChoice
from .data_manager import DataManager
from .oracles_manager import OraclesManager
from .config import BundleChoiceConfig, DimensionsConfig
__all__ = ['BundleChoice', 'DataManager', 'OraclesManager', 'BundleChoiceConfig', 'DimensionsConfig', '__version__', '__version_info__']