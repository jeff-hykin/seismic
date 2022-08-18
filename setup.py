import setuptools
from setuptools import find_packages
with open("README.md", "r") as fh:
    long_description = fh.read()
neuprint_reqs = ['neuprint-python']
plotting_reqs = ['matplotlib','tensorboard', 'jupyter','notebook']
setuptools.setup(
    name="neuroaiengines",
    version="1.0.0",
    author="Raphael Norman-Tenazas",
    author_email="raphael.norman-tenazas@jhuapl.edu",
    description="Runs NeuroAIEngines models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aplbrain/neuroaiengines-model",
    packages=find_packages('.'),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'scipy',
        'torch',
        'networkx',
        'pandas',
        'cloudpickle',
        'namegenerator',
        'tqdm',
        
    ],
    package_data={"": ["*.csv","*.pkl"]},
    extras_require={
        
        'neuprint':neuprint_reqs,
        'plotting':plotting_reqs,
        'all':neuprint_reqs+plotting_reqs,
    },
    python_requires='>=3.6'
)