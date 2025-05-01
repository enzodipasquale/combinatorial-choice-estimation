from setuptools import setup, find_packages

setup(
    name='bundlechoice',
    version='0.1.0',
    description='Estimation toolkit for bundled discrete choice models',
    author='Enzo Di Pasquale',
    author_email='ed2189@nyu.edu',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'gurobipy',
        # 'mpi4py',
        'pyyaml',
        'matplotlib',
    ],
    python_requires='>=3.9',
)
