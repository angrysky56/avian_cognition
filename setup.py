"""
Setup script for Avian Cognitive Architecture package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="avian_cognition",
    version="0.1.0",
    author="angrysky56",
    author_email="youremail@example.com",
    description="Neural architectural framework inspired by avian cognition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/angrysky56/avian_cognition",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
)
