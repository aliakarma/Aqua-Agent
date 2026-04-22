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
        # FIX-17 (R1-mn4 / R2-mn4): Pinned to match requirements.txt exactly.
        # torchvision removed (never imported in src/).
        "torch>=2.2.0",
        "numpy>=1.26.4",
        "scipy>=1.12.0",
        "pyyaml>=6.0.1",
        "h5py>=3.10.0",
        "pandas>=2.2.0",
        "scikit-learn>=1.4.0",
        "tensorboard>=2.16.2",
        "epyt>=1.1.4",
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
