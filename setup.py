from setuptools import setup, find_packages

# Use pyproject.toml for configuration
# This file exists for backward compatibility
# Version is read from bundlechoice/__init__.py to maintain single source of truth
def _get_version():
    """Read version from package __init__.py."""
    import os
    version_file = os.path.join(os.path.dirname(__file__), 'bundlechoice', '__init__.py')
    with open(version_file, 'r') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"').strip("'")
    raise RuntimeError("Unable to find version string")

setup(
    name='bundlechoice',
    version=_get_version(),
    description='Estimation toolkit for combinatorial discrete choice models',
    author='Enzo Di Pasquale',
    author_email='ed2189@nyu.edu',
    packages=find_packages(exclude=['bundlechoice._legacy*', 'bundlechoice.tests*']),
    install_requires=[
        'numpy>=1.24',
        'pandas>=2.0',
        'scipy>=1.10',
        'gurobipy>=11.0',
        'mpi4py>=3.1',
        'pyyaml>=6.0',
        'matplotlib>=3.7',
        'networkx>=3.0',
    ],
    python_requires='>=3.9',
    license='MIT',
)
