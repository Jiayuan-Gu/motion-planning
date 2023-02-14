# Pythonic Motion Planning (pymp)

[![PyPI version](https://badge.fury.io/py/motion-planning.svg)](https://badge.fury.io/py/motion-planning)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jiayuan-Gu/pymp/blob/main/example.ipynb)

> [Motion planning](https://en.wikipedia.org/wiki/Motion_planning) is a computational problem to find a sequence of valid configurations that moves the object from the source to destination.

This library is designed for robotic applications, especially sampling-based algorithms for high-dimension configuration spaces(e.g., robot arm).

- pythonic: easy to debug, customize and extend
- standalone collision checker ([hpp-fcl](https://github.com/humanoid-path-planner/hpp-fcl)): without relying on any physical simulator (e.g., mujoco, pybullet, sapien) to check collision
- out-of-box: common motion planning algorithms (e.g., RRT-Connect) are implemented for robotic manipulation

## Installation

### Dependencies

This library (*pymp*) depends on *[pinocchio](https://github.com/stack-of-tasks/pinocchio)* to handle URDF and robot kinematics, *[hpp-fcl](https://github.com/humanoid-path-planner/hpp-fcl)* to check collision, *[toppra](https://github.com/hungpham2511/toppra)* to do time parameterization.

### Install pymp

From pip:

```bash
pip install motion-planning
```

From source:

```bash
git clone https://github.com/Jiayuan-Gu/pymp.git
pip install -e .
```

From Github directly:

```bash
pip install --upgrade git+https://github.com/Jiayuan-Gu/pymp.git
```

## Usage

See [example.py](example.py) for basic usage. Note that `pymp` depends on SRDF associated with URDF to remove self-collision.

### Logging

The logging level can be specified by the environment variable `PYMP_LOG`.

```bash
# Set the logging level to DEBUG for pymp
export PYMP_LOG=DEBUG
```

### Base pose

`pymp` supports specifying the pose of the base link during the initialization of the planner. We support many formats of pose (e.g., \[x, y, z\] for position, \[w, i, j, k\] for quaternion, \[x, y, z, w, i, j, k\] for SE(3), or a 4x4 rigid transformation matrix)

```python
from pymp import Planner

planner = Planner(
    ...
    base_pose=[0, 0, 0],
)
```

## Troubleshooting

- `ImportError: libboost_python38.so`: try to force reinstall pinocchio, e.g., `pip install pin --no-cache-dir --force-reinstall --upgrade`.
