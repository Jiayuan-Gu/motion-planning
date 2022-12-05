# Pythonic Motion Planning (pymp)

## Installation

### Dependencies

*pymp* depends on *[pinocchio](https://github.com/stack-of-tasks/pinocchio)* to handle URDF and robot kinematics, *[hpp-fcl](https://github.com/humanoid-path-planner/hpp-fcl)* to check collision, *[toppra](https://github.com/hungpham2511/toppra)* to do time parameterization.

For `pinocchio` and `hpp-fcl`, their pip wheels require glibc>=2.28 (ubuntu>=20.04). If your OS system is not supported, please install them through conda instead. Thus, we do not add them as requirements in `setup.py`.

```bash
pip install "pin>=2.6.12"
# If not supported, use conda instead
# conda install "pinocchio>=2.6.12" -c conda-forge
```

### Install pymp

From source:

```bash
git clone https://github.com/Jiayuan-Gu/pymp.git
python setup.py install
```

From Github directly:

```bash
pip install --upgrade git+https://github.com/Jiayuan-Gu/pymp.git
```

## Usage

See [example.py](example.py) for basic usage.

**Logging**

The logging level can be specified by the environment variable `PYMP_LOG` (e.g., `PYMP_LOG=DEBUG`).

**Base position**

`pymp` supports specifying the pose of the base link during the initialization of the planner.

**Self-collision**

`pymp` depends on SRDF associated with URDF to remove self-collision.
