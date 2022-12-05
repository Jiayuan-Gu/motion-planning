# Copyright (c) 2022 Jiayuan Gu
# Licensed under The MIT License [see LICENSE for details]

import numpy as np
import pinocchio


def xyz_wijk_to_SE3(xyz, wijk):
    pos = np.array(xyz)
    # NOTE(jigu): pinocchio uses XYZW format for quaternions
    quat = np.array(wijk)[[1, 2, 3, 0]]
    return pinocchio.SE3(pinocchio.Quaternion(quat), pos)


def toSE3(x):
    """Convert to pinocchio.SE3

    Args:
        x: input pose. See notes for supported formats.

    Returns:
        pinocchio.SE3

    Notes:
        We support the following formats:
        - xyz: [3] for position
        - wijk: [4] for quaternion
        - xyzwijk: [7] for position and quaternion
        - T: [4, 4], rigid transformation
        - sapien: sapien.Pose, which has p and q
    """
    if isinstance(x, pinocchio.SE3):
        return x
    if isinstance(x, (tuple, list)) and len(x) == 2:
        pose = xyz_wijk_to_SE3(*x)
    elif x.__class__.__name__ == "Pose":  # sapien.Pose
        pose = xyz_wijk_to_SE3(x.p, x.q)
    elif np.ndim(x) == 1:
        if len(x) == 3:
            pose = xyz_wijk_to_SE3(x, [1, 0, 0, 0])
        elif len(x) == 4:
            pose = xyz_wijk_to_SE3([0, 0, 0], x)
        elif len(x) == 7:
            pose = xyz_wijk_to_SE3(x[0:3], x[3:])
        else:
            raise RuntimeError(x)
    elif np.ndim(x) == 2:
        pose = pinocchio.SE3(np.array(x))
    else:
        raise RuntimeError(x)
    return pose
