# Copyright (c) 2022 Jiayuan Gu
# Licensed under The MIT License [see LICENSE for details]

import io
import logging
import os
from typing import Union

import hppfcl as fcl
import numpy as np
import pinocchio as pin
import trimesh
import trimesh.convex
from bs4 import BeautifulSoup

from pymp.utils import toSE3

logger = logging.getLogger("pymp.robot")


def cook_urdf_for_pinocchio(urdf_path, use_convex, package_dir: str = None):
    with open(urdf_path, "r") as f:
        urdf_xml = BeautifulSoup(f.read(), "xml")

    # for mesh_tag in urdf_xml.find_all("mesh"):
    for mesh_tag in urdf_xml.select("collision mesh"):
        filename = mesh_tag["filename"].lstrip("package://")

        # Use convex collision shape
        if use_convex:
            assert (
                package_dir is not None
            ), "package_dir must be specified if using convex collision"
            # First check if the (SAPIEN) convex hull file exists
            convex_path = os.path.join(package_dir, filename + ".convex.stl")
            if os.path.exists(convex_path):
                filename = filename + ".convex.stl"
            else:
                # Then check if the (trimesh) convex hull file exists
                convex2_path = os.path.join(package_dir, filename + ".convex2.stl")
                if not os.path.exists(convex2_path):
                    logger.info(
                        "Convex hull ({}) not found, generating...".format(convex2_path)
                    )
                    mesh_path = os.path.join(package_dir, filename)
                    mesh = trimesh.load_mesh(mesh_path)
                    cvx_mesh = trimesh.convex.convex_hull(mesh)
                    cvx_mesh.export(convex2_path)
                filename = filename + ".convex2.stl"

        # Add "package://" for Pinocchio
        mesh_tag["filename"] = "package://" + filename

    return urdf_xml


def load_model_from_urdf(
    urdf_path, load_collision=True, use_convex=False, floating=False
):
    """Load a Pinocchio model from URDF."""
    # model = pin.buildModelFromUrdf(urdf_path)

    package_dir = os.path.dirname(urdf_path)
    urdf_xml = cook_urdf_for_pinocchio(urdf_path, use_convex, package_dir=package_dir)
    urdf_str = str(urdf_xml)
    urdf_stream = io.StringIO(urdf_str).read()
    if floating:
        model = pin.buildModelFromXML(urdf_stream, pin.JointModelFreeFlyer())
    else:
        model = pin.buildModelFromXML(urdf_stream)

    collision_model = None
    if load_collision:
        # Load collision geometries
        collision_model = pin.buildGeomFromUrdfString(
            model, urdf_str, pin.GeometryType.COLLISION, package_dirs=package_dir
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

    def getAdjacentPairs(self):
        adjacent_pairs = []
        for frame in self.model.frames:
            if frame.type != pin.FrameType.BODY:
                continue
            parent_frame = frame
            while True:
                parent_frame_idx = parent_frame.previousFrame
                parent_frame = self.model.frames[parent_frame_idx]
                if parent_frame_idx == 0 or parent_frame.type == pin.FrameType.BODY:
                    break
            if parent_frame.name != "universe":
                adjacent_pairs.append(
                    dict(link1=parent_frame.name, link2=frame.name, reason="Adjacent")
                )
        return adjacent_pairs

    def findAlwaysCollisionPairs(self, n=1000):
        # Find the link of each geometry
        geoms = self.collision_model.geometryObjects
        geom_link_names = []
        for i in range(self.collision_model.ngeoms):
            geom = geoms[i]
            link_name = "_".join(geom.name.split("_")[:-1])
            geom_link_names.append(link_name)

        link_names = list(set(geom_link_names))
        n_links = len(link_names)
        link_name_to_id = {name: i for i, name in enumerate(link_names)}
        geom_link_ids = [link_name_to_id[x] for x in geom_link_names]

        # Matrix to count the number of collisions between each pair of links
        count = np.zeros([n_links, n_links], dtype=int)

        for _ in range(n):
            q = pin.randomConfiguration(self.model)

            # Compute all the collisions
            data = self.data
            collision_data = self.collision_data
            is_collided = pin.computeCollisions(
                self.model, data, self.collision_model, collision_data, q, False
            )

            if not is_collided:
                break

            cmat = np.zeros([n_links, n_links], dtype=bool)

            # According to examples/collisions.py in pinocchio
            n_cp = len(self.collision_model.collisionPairs)
            for i in range(n_cp):
                cp = self.collision_model.collisionPairs[i]
                cr = collision_data.collisionResults[i]
                if cr.isCollision():
                    idx1 = geom_link_ids[cp.first]
                    idx2 = geom_link_ids[cp.second]
                    cmat[idx1][idx2] = 1

            # Update the count matrix
            count = count + cmat

        collision_pairs = []
        for i in range(n_links):
            for j in range(n_links):
                if count[i][j] < n:
                    continue
                link1 = link_names[i]
                link2 = link_names[j]
                logger.info(f"{link1} always collides with {link2}")
                collision_pairs.append(dict(link1=link1, link2=link2))

        return collision_pairs

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
