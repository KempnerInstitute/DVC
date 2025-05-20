from setuptools import setup, find_packages

setup(
    name='DVC',
    version='0.1.0',
    description='Multivariate vine-copula densities, entropies and other probabilistic metrics using parametric and non-parametric methods.',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'torch',
        'scipy',
        'pyyaml',
        'matplotlib',
        'networkx',
    ],
    python_requires='>=3.8',
) 