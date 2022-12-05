# Copyright (c) 2022 Jiayuan Gu
# Licensed under The MIT License [see LICENSE for details]

import io
import logging
import os
from typing import Union

import hppfcl as fcl
import numpy as np
import pinocchio as pin
from bs4 import BeautifulSoup

from pymp.utils import toSE3

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


def load_model_from_urdf(
    urdf_path, load_collision=True, use_convex=False, floating=False
):
    """Load a Pinocchio model from URDF."""
    # model = pin.buildModelFromUrdf(urdf_path)
    urdf_str = cook_urdf_for_pinocchio(urdf_path, use_convex)
    urdf_stream = io.StringIO(urdf_str).read()
    if floating:
        model = pin.buildModelFromXML(urdf_stream, pin.JointModelFreeFlyer())
    else:
        model = pin.buildModelFromXML(urdf_stream)

    collision_model = None
    if load_collision:
        # Load collision geometries
        mesh_dir = os.path.dirname(urdf_path)
        collision_model = pin.buildGeomFromUrdfString(
            model, urdf_str, pin.GeometryType.COLLISION, package_dirs=mesh_dir
        )

    return model, collision_model


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
    oMdes = pin.SE3(T)

    if init_q is None:
        q = pin.neutral(model)
    else:
        q = np.array(init_q)

    success = False
    damp_I = damp * np.eye(6)
    for _ in range(max_iters):
        pin.forwardKinematics(model, data, q)
        dMi = oMdes.actInv(data.oMi[joint_id])
        err = pin.log(dMi).vector
        if np.linalg.norm(err) < eps:
            success = True
            break
        J = pin.computeJointJacobian(model, data, q, joint_id)
        v = -J.T.dot(np.linalg.solve(J.dot(J.T) + damp_I, err))
        q = pin.integrate(model, q, v * dt)

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
    qmask=None,
):
    """Closed-Loop Inverse Kinematics (frame)."""
    data = model.createData()
    oMdes = pin.SE3(T)

    if init_q is None:
        q = pin.neutral(model)
    else:
        q = np.array(init_q)

    success = False
    damp_I = damp * np.eye(6)
    for _ in range(max_iters):
        pin.forwardKinematics(model, data, q)
        oMf = pin.updateFramePlacement(model, data, frame_id)
        dMf = oMdes.actInv(oMf)
        err = pin.log(dMf).vector
        if np.linalg.norm(err) < eps:
            success = True
            break
        J = pin.computeFrameJacobian(model, data, q, frame_id)  # [6, nq]
        if qmask is not None:
            J[:, qmask] = 0
        v = -J.T.dot(np.linalg.solve(J.dot(J.T) + damp_I, err))
        q = pin.integrate(model, q, v * dt)

    return q, success, err


def makeConvex(vertices, faces):
    vs = fcl.StdVec_Vec3f()
    vs.extend(vertices)
    fs = fcl.StdVec_Triangle()
    for face in faces:
        fs.append(fcl.Triangle(*face.tolist()))
    return fcl.Convex(vs, fs)


class RobotWrapper(pin.RobotWrapper):
    def __init__(self, *args, base_pose=None, **kwargs):
        super().__init__(*args, **kwargs)

        if base_pose is None:
            base_pose = pin.SE3.Identity()
        self._base_pose = base_pose

    @classmethod
    def loadFromURDF(
        cls, filename, load_collision=True, use_convex=True, floating=False, **kwargs
    ):
        # TODO(jigu): do we need to get visual model?
        model, collision_model = load_model_from_urdf(
            filename,
            load_collision=load_collision,
            use_convex=use_convex,
            floating=floating,
        )
        return cls(
            model=model,
            collision_model=collision_model,
            visual_model=collision_model,
            **kwargs,
        )

    def buildReducedRobot(self, list_of_joints_to_lock, reference_configuration=None):
        # if joint to lock is a string, try to find its index
        lockjoints_idx = []
        for jnt in list_of_joints_to_lock:
            idx = jnt
            if isinstance(jnt, str):
                idx = self.model.getJointId(jnt)
            lockjoints_idx.append(idx)

        if reference_configuration is None:
            reference_configuration = pin.neutral(self.model)

        model, geom_models = pin.buildReducedModel(
            model=self.model,
            list_of_geom_models=[self.visual_model, self.collision_model],
            list_of_joints_to_lock=lockjoints_idx,
            reference_configuration=reference_configuration,
        )

        return RobotWrapper(
            model=model,
            collision_model=geom_models[0],
            visual_model=geom_models[0],  # reuse collision for visual
            base_pose=self._base_pose,  # initialize base pose
        )

    @property
    def base_pose(self):
        return np.array(self._base_pose)

    # TODO(jigu): set base pose (need to sync all geometries)

    # -------------------------------------------------------------------------- #
    # Shortcuts to Pinocchio
    # -------------------------------------------------------------------------- #
    @property
    def joint_names(self):
        return list(self.model.names)

    @property
    def active_joint_names(self):
        nqs = self.model.nqs
        return [name for i, name in enumerate(self.model.names) if nqs[i] > 0]

    @property
    def joint_limits(self):
        return self.model.lowerPositionLimit, self.model.upperPositionLimit

    def within_joint_limits(self, q, eps=1e-4, return_mask=False):
        assert len(q) == self.model.nq, (len(q), self.model.nq)
        lower, upper = self.model.lowerPositionLimit, self.model.upperPositionLimit
        mask = np.logical_and(q >= (lower - eps), q <= (upper + eps))
        if return_mask:
            return mask
        else:
            return np.all(mask).item()

    def joint_index(self, name):
        return self.model.getJointId(name)

    def joint_nq(self, name):
        return self.model.nqs[self.joint_index(name)]

    def joint_idx_q(self, name):
        return self.model.idx_qs[self.joint_index(name)]

    def frame_index(self, name):
        return self.model.getFrameId(name)

    def get_frame(self, index: Union[int, str]):
        if isinstance(index, str):
            index = self.frame_index(index)
        return self.model.frames[index]

    def get_support_joint_ids(self, link_name, exclude_inactive=True):
        frame = self.get_frame(link_name)
        support_joint_ids = list(self.model.supports[frame.parent])
        if exclude_inactive:
            nqs = self.model.nqs
            support_joint_ids = [i for i in support_joint_ids if nqs[i] > 0]
        return support_joint_ids

    def computeFrameJacobianWorld(self, q, frame_id):
        # The default reference frame is local.
        return pin.computeFrameJacobian(
            self.model, self.data, q, frame_id, pin.ReferenceFrame.WORLD
        )

    def framePlacement(self, q, index: Union[int, str], update_kinematics=True):
        if update_kinematics:
            pin.forwardKinematics(self.model, self.data, q)
        if isinstance(index, str):
            index = self.model.getFrameId(index)
        return pin.updateFramePlacement(self.model, self.data, index)

    # -------------------------------------------------------------------------- #
    # Algorithms
    # -------------------------------------------------------------------------- #
    def compute_CLIK(
        self,
        link2base,
        link_name,
        init_q=None,
        max_iters=1000,
        eps=1e-4,
        dt=1e-1,
        damp=1e-12,
        qmask=None,
        measurement="frame_pose",
    ):
        link2base = self._base_pose.inverse() * link2base
        if measurement == "frame_pose":
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
                qmask=qmask,
            )
        elif measurement == "joint_pose":
            frame_id = self.model.getFrameId(link_name)
            frame = self.model.frames[frame_id]
            link2joint = frame.placement
            joint_id = frame.parent
            link2base = pin.SE3(link2base)

            # Transform frame pose to link pose
            joint2base = link2base * link2joint.inverse()

            return compute_CLIK_joint(
                self.model,
                joint2base,
                joint_id,
                init_q=init_q,
                max_iters=max_iters,
                eps=eps,
                dt=dt,
                damp=damp,
            )
        else:
            raise NotImplementedError(measurement)

    # ---------------------------------------------------------------------------- #
    # Collision
    # ---------------------------------------------------------------------------- #
    def initCollisionPairs(self):
        """Initialize (robot) collision pairs.
        It should be called before loading geometries other than the robot.
        """
        self.collision_model.addAllCollisionPairs()
        logger.debug(
            "num collision pairs - initial: %d",
            len(self.collision_model.collisionPairs),
        )

        # Initialize all link geometries
        self._link_collision_ids = []
        for go in self.collision_model.geometryObjects:
            collision_id = self.collision_model.getGeometryId(go.name)
            self._link_collision_ids.append(collision_id)
            if go.geometry.getObjectType() == fcl.OT_BVH:
                # NOTE(jigu): It does not compute a convex hull. Just enable `.convex`
                go.geometry.buildConvexRepresentation(False)
                go.geometry = go.geometry.convex
        logger.debug("num collision links: %d", len(self._link_collision_ids))

        self.rebuildData()

    def removeCollisionPairsFromSRDF(self, srdf_path, verbose=False):
        """Remove collision pairs listed in the SRDF file."""
        pin.removeCollisionPairs(
            self.model, self.collision_model, srdf_path, verbose=verbose
        )
        logger.debug(
            "num collision pairs - after removing useless collision pairs: %d",
            len(self.collision_model.collisionPairs),
        )
        self.rebuildData()

    def computeCollisions(self, q):
        # NOTE(jigu): https://github.com/stack-of-tasks/pinocchio/issues/1701
        # CAUTION: remember to rebuild data when we modify model!

        # Create data structures
        # data = self.model.createData()
        # collision_data = pin.GeometryData(self.collision_model)
        data = self.data
        collision_data = self.collision_data

        # Compute all the collisions
        return pin.computeCollisions(
            self.model,
            data,
            self.collision_model,
            collision_data,
            np.array(q),
            stop_at_first_collision=True,
        )

    def isCollisionFree(self, q):
        return not self.computeCollisions(q)

    def addCollisionPairs(self, collision_id):
        """Add collision pairs between the input and all existing geometries."""
        gos = self.collision_model.geometryObjects
        go = gos[collision_id]
        for i in range(self.collision_model.ngeoms):
            if collision_id == i:
                continue
            # Ignore collision with the same joint
            if gos[i].parentJoint == go.parentJoint:
                logger.debug("Ignore collision between %s and %s", go.name, gos[i].name)
                continue
            collision_pair = pin.CollisionPair(collision_id, i)
            self.collision_model.addCollisionPair(collision_pair)
        self.rebuildData()

    def disableCollision(self, index: Union[int, str], flag=True):
        if isinstance(index, str):
            index = self.collision_model.getGeometryId(index)
        self.collision_model.geometryObjects[index].disableCollision = flag
        self.rebuildData()

    def getGeometry(self, index: Union[int, str]):
        if isinstance(index, str):
            index = self.collision_model.getGeometryId(index)
        return self.collision_model.geometryObjects[index]

    def removeGeometry(self, name):
        self.collision_model.removeGeometryObject(name)
        self.rebuildData()

    def addGeometry(
        self, name, geometry, pose, parent_joint=0, color=None, add_collision=True
    ):
        # https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/structpinocchio_1_1GeometryObject.html
        pose = pin.SE3.Identity() if pose is None else toSE3(pose)

        # Convert to robot base frame
        pose = self._base_pose.inverse() * pose

        go = pin.GeometryObject(name, parent_joint, geometry, pose)
        if color is not None:
            go.meshColor = np.array(color)

        # NOTE(jigu): pinocchio can take multiple names.
        # But getGeometryId only returns the first one
        if self.collision_model.existGeometryName(name):
            # logger.warn("'%s' has existed in the collision model", name)
            raise RuntimeError(f"{name} has existed in the collision model")

        if add_collision:
            collision_id = self.collision_model.addGeometryObject(go)
            logger.debug("%s(%d) is added to collision model", name, collision_id)
            self.addCollisionPairs(collision_id)
        return go

    def addBox(self, size, pose=None, color=(0, 1, 0, 1), name="box"):
        """Add a box to the collision model.

        Args:
            size (tuple, np.ndarray): full size, with shape [3]
            pose (pin.SE3, np.ndarray, optional): SE3 transformation. If None, set to Identity.
            color (tuple, optional): color to visualize.
            name (str, optional): name of object.
        """
        if isinstance(size, np.ndarray):
            size = size.tolist()
        box = fcl.Box(*size)
        self.addGeometry(name, box, pose, color=color)

    def attachBox(
        self, size, pose, frame_index, color=(1, 1, 0, 1), name="attached_box"
    ):
        """Attach a box to the collision model.

        Args:
            size (tuple, np.ndarray): full size, with shape [3]
            pose (pin.SE3, np.ndarray): pose relative to the attached frame.
            frame_index (int, str): index or name of attached frame.
            color (tuple, optional): color to visualize.
            name (str, optional): name of object.
        """
        box = fcl.Box(*size)
        frame = self.get_frame(frame_index)
        pose = frame.placement * toSE3(pose)
        self.addGeometry(name, box, pose, parent_joint=frame.parent, color=color)

    def addOctree(self, points, resolution, pose=None, name="octree"):
        """Add an octree of point cloud to the collision model.

        Args:
            points (np.ndarray): point cloud
            resolution (float): octree resolution
            pose (pin.SE3, np.ndarray, optional): pose of point cloud
            name (str, optional): name of object.
        """
        octree = fcl.makeOctree(points, resolution)
        # TODO(jigu): need to verify whether octree can have pose
        self.addGeometry(name, octree, pose)

    def addPointCloudVisual(self, points, pose=None, name="point_cloud"):
        pcd = fcl.BVHModelOBBRSS()
        pcd.beginModel(0, len(points))
        pcd.addVertices(points)
        pcd.endModel()
        go = self.addGeometry(name, pcd, pose, False)
        # NOTE(jigu): Currently visual_model == collision_model
        self.visual_model.addGeometryObject(go)
        self.rebuildData()

    def addConvex(self, vertices, faces, pose=None, name="convex"):
        # NOTE(jigu): Multiple convex shapes need to be added individually
        convex = makeConvex(vertices, faces)
        go = pin.GeometryObject(name, convex, pose)
        self.addGeometry(name, go, pose)

    def addMeshVisual(self, vertices, faces, pose=None, name="mesh"):
        model = fcl.BVHModelOBBRSS()
        model.beginModel(0, len(vertices))
        model.addVertices(vertices)
        model.addTriangles(faces)
        model.endModel()
        go = self.addGeometry(name, model, pose, False)
        # NOTE(jigu): Currently visual_model == collision_model
        self.visual_model.addGeometryObject(go)
        self.rebuildData()
