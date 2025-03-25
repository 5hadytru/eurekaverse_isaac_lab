from setuptools import find_packages
from distutils.core import setup

setup(
    name='eurekaverse_isaac_lab',
    author='',
    version='1.0',
    description='',
    python_requires='>=3.10',
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'openai',
        'opencv-python',
        'pydelatin',
        'pyfqmr',
        'hydra-core',
        'wandb',
        'gpustat',
        'tqdm',
        'ipdb',
    ],
    packages=find_packages()
)
