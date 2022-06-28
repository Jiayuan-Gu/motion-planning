import numpy as np
import pinocchio


def xyz_wijk_to_SE3(xyz, wijk):
    pos = np.array(xyz)
    # NOTE(jigu): pinocchio uses XYZW format for quaternions
    quat = np.array(wijk)[[1, 2, 3, 0]]
    return pinocchio.SE3(pinocchio.Quaternion(quat), pos)
