import io
import logging
import os

import hppfcl as fcl
import numpy as np
import pinocchio
from bs4 import BeautifulSoup

logger = logging.getLogger("pymp.robot")


def cook_urdf_for_pinocchio(urdf_path, use_convex):
    with open(urdf_path, "r") as f:
        urdf_xml = BeautifulSoup(f.read(), "xml")

    # Add "package://" for Pinocchio
    # for mesh_tag in urdf_xml.find_all("mesh"):
    for mesh_tag in urdf_xml.select("collision mesh"):
        if not mesh_tag["filename"].startswith("package://"):
            mesh_tag["filename"] = "package://" + mesh_tag["filename"]
        if use_convex:
            mesh_tag["filename"] = mesh_tag["filename"] + ".convex.stl"

    return str(urdf_xml)


def load_model_from_urdf(urdf_path, load_collision=True, use_convex=False):
    """Load a Pinocchio model from URDF."""
    # model = pinocchio.buildModelFromUrdf(urdf_path)
    urdf_str = cook_urdf_for_pinocchio(urdf_path, use_convex)
    urdf_stream = io.StringIO(urdf_str).read()
    model = pinocchio.buildModelFromXML(urdf_stream)

    if load_collision:
        # Load collision geometries
        mesh_dir = os.path.dirname(urdf_path)
        collision_model = pinocchio.buildGeomFromUrdfString(
            model, urdf_str, pinocchio.GeometryType.COLLISION, package_dirs=mesh_dir
        )
        return model, collision_model
    else:
        return model


def compute_CLIK_joint(
    model,
    T,
    joint_id: int,
    init_q=None,
    max_iters=1000,
    eps=1e-4,
    dt=1e-1,
    damp=1e-12,
):
    """Closed-Loop Inverse Kinematics (joint)."""
    data = model.createData()
    oMdes = pinocchio.SE3(T)

    if init_q is None:
        q = pinocchio.neutral(model)
    else:
        q = np.array(init_q)

    success = False
    damp_I = damp * np.eye(6)
    for _ in range(max_iters):
        pinocchio.forwardKinematics(model, data, q)
        dMi = oMdes.actInv(data.oMi[joint_id])
        err = pinocchio.log(dMi).vector
        if np.linalg.norm(err) < eps:
            success = True
            break
        J = pinocchio.computeJointJacobian(model, data, q, joint_id)
        v = -J.T.dot(np.linalg.solve(J.dot(J.T) + damp_I, err))
        q = pinocchio.integrate(model, q, v * dt)

    return q, success, err


def compute_CLIK_frame(
    model,
    T,
    frame_id: int,
    init_q=None,
    max_iters=1000,
    eps=1e-4,
    dt=1e-1,
    damp=1e-12,
):
    """Closed-Loop Inverse Kinematics (frame)."""
    data = model.createData()
    oMdes = pinocchio.SE3(T)

    if init_q is None:
        q = pinocchio.neutral(model)
    else:
        q = np.array(init_q)

    success = False
    damp_I = damp * np.eye(6)
    for _ in range(max_iters):
        pinocchio.forwardKinematics(model, data, q)
        oMf = pinocchio.updateFramePlacement(model, data, frame_id)
        dMf = oMdes.actInv(oMf)
        err = pinocchio.log(dMf).vector
        if np.linalg.norm(err) < eps:
            success = True
            break
        J = pinocchio.computeFrameJacobian(model, data, q, frame_id)
        v = -J.T.dot(np.linalg.solve(J.dot(J.T) + damp_I, err))
        q = pinocchio.integrate(model, q, v * dt)

    return q, success, err


class RobotWrapper(pinocchio.RobotWrapper):
    @classmethod
    def loadFromURDF(cls, filename, load_collision=True, use_convex=True):
        model, collision_model = load_model_from_urdf(
            filename, load_collision=load_collision, use_convex=use_convex
        )
        return cls(
            model=model, collision_model=collision_model, visual_model=collision_model
        )

    @property
    def joint_limits(self):
        return self.model.lowerPositionLimit, self.model.upperPositionLimit

    def compute_CLIK(
        self,
        link2base,
        link_name,
        init_q=None,
        max_iters=1000,
        eps=1e-4,
        dt=1e-1,
        damp=1e-12,
    ):
        frame_id = self.model.getFrameId(link_name)
        return compute_CLIK_frame(
            self.model,
            link2base,
            frame_id,
            init_q=init_q,
            max_iters=max_iters,
            eps=eps,
            dt=dt,
            damp=damp,
        )

        # ---------------------------------------------------------------------------- #
        # Joint frame IK
        # ---------------------------------------------------------------------------- #
        # frame_id = self.model.getFrameId(link_name)
        # frame = self.model.frames[frame_id]
        # link2joint = frame.placement
        # joint_id = frame.parent
        # link2base = pinocchio.SE3(link2base)

        # # Transform frame pose to link pose
        # joint2base = link2base * link2joint.inverse()

        # return compute_CLIK_joint(
        #     self.model,
        #     joint2base,
        #     joint_id,
        #     init_q=init_q,
        #     max_iters=max_iters,
        #     eps=eps,
        #     dt=dt,
        #     damp=damp,
        # )
        # ---------------------------------------------------------------------------- #

    # ---------------------------------------------------------------------------- #
    # Collision
    # ---------------------------------------------------------------------------- #
    def initCollisionPairs(self):
        self.collision_model.addAllCollisionPairs()
        logger.debug(
            "num collision pairs - initial: %d",
            len(self.collision_model.collisionPairs),
        )

        # Record all link geometries
        self._link_collision_ids = []
        for go in self.collision_model.geometryObjects:
            collision_id = self.collision_model.getGeometryId(go.name)
            self._link_collision_ids.append(collision_id)
            if go.geometry.getObjectType() == fcl.OT_BVH:
                # NOTE(jigu): It does not compute a convex hull. Just enable `.convex`
                go.geometry.buildConvexRepresentation(False)
                go.geometry = go.geometry.convex

    def removeCollisionPairsFromSRDF(self, srdf_path, verbose=False):
        """Remove collision pairs listed in the SRDF file."""
        pinocchio.removeCollisionPairs(
            self.model, self.collision_model, srdf_path, verbose=verbose
        )
        logger.debug(
            "num collision pairs - after removing useless collision pairs: %d",
            len(self.collision_model.collisionPairs),
        )

    def computeCollisions(self, q):
        # Create data structures
        data = self.model.createData()
        geom_data = pinocchio.GeometryData(self.collision_model)

        # Compute all the collisions
        return pinocchio.computeCollisions(
            self.model,
            data,
            self.collision_model,
            geom_data,
            np.array(q),
            stop_at_first_collision=True,
        )

    def isCollisionFree(self, q):
        return not self.computeCollisions(q)

    def addBox(self, size, pose, color=(0, 1, 0, 1)):
        box = fcl.Box(*size)
        go_box = pinocchio.GeometryObject("box", 0, box, pinocchio.SE3(pose))
        go_box.meshColor = np.array(color)

        box_collision_id = self.collision_model.addGeometryObject(go_box)
        for link_collision_id in self._link_collision_ids:
            collision_pair = pinocchio.CollisionPair(
                box_collision_id, link_collision_id
            )
            self.collision_model.addCollisionPair(collision_pair)
