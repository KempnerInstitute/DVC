from setuptools import setup, find_packages

setup(
    name='DVC',
    version='0.1.0',
    description='Multivariate vine-copula densities, entropies and other probabilistic metrics.',
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
        # add pytest here if you want it installed by default:
        'pytest',
    ],
    # or, if you’d rather only pull pytest in for testing:
    tests_require=[
        'pytest',
    ],
    extras_require={
        'dev': [
            'pytest',
            # any other dev-only tools…
        ]
    },
    python_requires='>=3.8',
)