"""
Setup configuration for crypto15 package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    with open(readme_file, "r", encoding="utf-8") as f:
        long_description = f.read()

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file, "r", encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="crypto15",
    version="0.1.0",
    author="Crypto Bot Team",
    description="A cryptocurrency trading bot framework for Kalshi",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DGator86/The-Crypto-Bot-for-Kalshi",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    entry_points={
        "console_scripts": [
            "crypto15-fetch=scripts.fetch_history:main",
            "crypto15-backtest=scripts.walkforward_backtest:main",
            "crypto15-train=scripts.train_full_and_save:main",
            "crypto15-predict=scripts.live_predict:main",
        ],
    },
)
