from setuptools import setup, find_packages

# Use pyproject.toml for configuration
# This file exists for backward compatibility
setup(
    name='bundlechoice',
    version='0.2.0',
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
