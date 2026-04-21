"""setup.py — AquaAgent package installation."""

from setuptools import setup, find_packages
from pathlib import Path

long_description = (Path(__file__).parent / "README.md").read_text()

setup(
    name="aquaagent",
    version="1.0.0",
    description="Agentic AI for Smart Water Distribution: Leak Detection and Governance",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="AquaAgent Authors",
    python_requires=">=3.10",
    packages=find_packages(include=["src", "src.*"]),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pyyaml>=6.0",
        "h5py>=3.8.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.2.0",
        "tensorboard>=2.12.0",
        "epyt>=1.0.0",
    ],
    extras_require={
        "graph": ["torch-geometric>=2.3.0"],
        "dev":   ["pytest", "pytest-cov", "black", "flake8", "isort"],
        "nb":    ["jupyter", "matplotlib", "seaborn"],
    },
    entry_points={
        "console_scripts": [
            "aquaagent-simulate=src.data.simulate:main",
            "aquaagent-train-ada=src.training.train_ada:main",
            "aquaagent-train-mappo=src.training.train_mappo:main",
            "aquaagent-evaluate=src.evaluation.evaluate:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Hydrology",
    ],
)
