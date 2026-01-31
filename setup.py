#!/usr/bin/env python3
"""
Pakit Setup Script
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme = Path(__file__).parent / "README.md"
long_description = readme.read_text(encoding="utf-8") if readme.exists() else ""

# Read requirements
requirements = Path(__file__).parent / "pakit_requirements.txt"
install_requires = []
if requirements.exists():
    install_requires = requirements.read_text().strip().split('\n')
    install_requires = [r.strip() for r in install_requires if r.strip() and not r.startswith('#')]

setup(
    name="pakit",
    version="1.0.0",
    description="Decentralized storage for BelizeChain with DAG, P2P, and ML optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="BelizeChain Team",
    author_email="development@belizechain.bz",
    url="https://github.com/BelizeChain/belizechain",
    license="MIT",
    packages=find_packages(exclude=["tests", "docs"]),
    python_requires=">=3.11",
    install_requires=install_requires,
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-asyncio>=0.23.0",
            "pytest-cov>=4.1.0",
            "black>=24.0.0",
            "flake8>=7.0.0",
            "mypy>=1.8.0",
        ],
        "ml": [
            "torch>=2.1.0",
            "scikit-learn>=1.4.0",
            "numpy>=1.26.0",
        ],
        "quantum": [
            "qiskit>=1.0.0",
            "qiskit-aer>=0.13.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pakit-server=api_server:main",
            "pakit-node=node.p2p_node:main",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: System :: Filesystems",
        "Topic :: Database :: Database Engines/Servers",
    ],
    keywords="blockchain storage decentralized dag p2p ipfs arweave",
    project_urls={
        "Documentation": "https://github.com/BelizeChain/pakit-storage/blob/main/README.md",
        "Source": "https://github.com/BelizeChain/pakit-storage",
        "Tracker": "https://github.com/BelizeChain/pakit-storage/issues",
    },
)
