# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
from pathlib import Path

# Get the long description from the README file
ROOT_DIR = Path(__file__).parent
long_description = (ROOT_DIR / "README.md").read_text(encoding="utf-8")

setup(
    name="motion_planning",
    version="0.1.5",
    author="Jiayuan Gu",
    author_email="jigu@ucsd.edu",
    keywords="robotics motion-planning",
    description="A pythonic motion planning library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Jiayuan-Gu/pymp",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "pin>=2.6.13",
        "toppra>=0.4.1",
        "lxml",
        "beautifulsoup4",
        "trimesh",
    ],
    extras_require={"tests": ["pytest", "black", "isort"], "meshcat": ["meshcat"]},
)

# python setup.py bdist_wheel
# twine upload dist/*
