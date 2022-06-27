# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(
    name="pymp",
    version="0.0.0",
    author_email="jigu@ucsd.edu",
    keywords="robotics motion planning",
    description="A lightweight motion planning library",
    classifiers=[
        "Operating System :: POSIX :: Linux",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Other Audience",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Framework :: Robot Framework :: Tool",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Education",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Utilities",
    ],
    packages=find_packages(include="pymp*"),
    python_requires=">=3.6",
    install_requires=["numpy", "pinocchio", "hppfcl", "toppra>=0.4.0", "pytransform3d"],
)
