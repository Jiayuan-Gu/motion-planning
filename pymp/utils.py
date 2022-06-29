import numpy as np
import pinocchio


def xyz_wijk_to_SE3(xyz, wijk):
    pos = np.array(xyz)
    # NOTE(jigu): pinocchio uses XYZW format for quaternions
    quat = np.array(wijk)[[1, 2, 3, 0]]
    return pinocchio.SE3(pinocchio.Quaternion(quat), pos)


def toSE3(x, fmt="xyzwijk"):
    """Convert to pinocchio.SE3

    Args:
        x: input pose. See notes for supported formats.
        fmt (str, optional): input format. Defaults to "xyzwijk".

    Returns:
        pinocchio.SE3

    Notes:
        We support the following formats:
        - xyzwijk: [7] or ([3], [4]) for position and quaternion
        - T: [4, 4], rigid transformation
        - sapien: sapien.Pose, which has p and q
    """
    if fmt == "xyzwijk":
        if len(x) == 2:  # x = (p, q)
            assert isinstance(x, (tuple, list)), type(x)
            pose = xyz_wijk_to_SE3(*x)
        elif len(x) == 3:
            pose = xyz_wijk_to_SE3(x, [1, 0, 0, 0])
        elif len(x) == 7:
            pose = xyz_wijk_to_SE3(x[0:3], x[3:])
        else:
            raise RuntimeError(x)
    elif fmt == "T":
        pose = pinocchio.SE3(np.array(x))
    elif fmt == "sapien":
        pose = xyz_wijk_to_SE3(x.p, x.q)
    else:
        raise NotImplementedError(fmt)
    return pose