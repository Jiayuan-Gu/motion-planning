# Copyright (c) 2022 Jiayuan Gu
# Licensed under The MIT License [see LICENSE for details]

import logging
import os
from typing import Sequence, Union

import numpy as np
import pinocchio as pin

from pymp.path_planning import GoalStates, JointStateSpace, RRTConnect
from pymp.robot import RobotWrapper
from pymp.utils import toSE3
from pymp.srdf_utils import dump_srdf

logger = logging.getLogger("pymp.planner")


class Planner:
    def __init__(
        self,
        urdf: str,
        user_joint_names: Sequence[str],
        ee_link_name: str,
        srdf: str = None,
        use_convex=True,
        planning_group: Sequence[str] = None,
        base_pose=None,
        timestep: float = None,
        joint_vel_limits: Union[float, Sequence[float], np.ndarray] = None,
        joint_acc_limits: Union[float, Sequence[float], np.ndarray] = None,
    ):
        r"""Motion planner for robots.

        Args:
            urdf: path to URDF (Unified Robot Description Format) file.
            user_joint_names: names of (active) joints.
                The order affects the order of input and output joint configurations.
                If None, use Pinocchio default order.
            ee_link_name: the end-effector link to plan.
            srdf: path to Semantic Robot Description Format file (SRDF).
                If None, try to replace the extension (*.urdf -> *.srdf).
            use_convex: whether to use convex collision shapes.
                If True, the *.convex.stl (generated by SAPIEN) is used.
            planning_group: names of joints to plan.
                If None, all supported joints of end-effector will be used.
            base_pose: pose of the robot base. If None, use identity.
            timestep: step for time parameterization. If None, disable time parameterization.
            joint_vel_limits: maximum joint velocities for time parameterization,
                which should have the same length as planning_group. Support broadcast.
            joint_acc_limits: maximum joint accelerations for time parameterization,
                which should have the same length as planning_group. Support broadcast.

        Raises:
            FileNotFoundError: SRDF is not found.

        See also:
            - https://moveit.picknik.ai/humble/doc/concepts/concepts.html
            - https://wiki.ros.org/urdf
            - https://wiki.ros.org/srdf
            - https://github.com/stack-of-tasks/pinocchio
            - https://github.com/humanoid-path-planner/hpp-fcl
            - https://github.com/ompl/ompl
            - https://github.com/hungpham2511/toppra
        """
        self.urdf = urdf
        if not srdf:
            srdf = urdf.replace(".urdf", ".srdf")
            logger.info("No SRDF provided. Use SRDF at {}.".format(srdf))
        self.srdf = srdf

        # Initialize Pinocchio model
        if base_pose is not None:
            base_pose = toSE3(base_pose)
        self.robot = RobotWrapper.loadFromURDF(
            urdf, use_convex=use_convex, base_pose=base_pose
        )

        self.robot.initCollisionPairs()
        if not os.path.exists(srdf):
            logger.info("SRDF ({}) not found. Generating...".format(srdf))
            adjacent_pairs = self.robot.getAdjacentPairs()
            always_pairs = self.robot.findAlwaysCollisionPairs(1000)
            dump_srdf(adjacent_pairs + always_pairs, srdf)
        self.robot.removeCollisionPairsFromSRDF(srdf)

        # Setup planning interface
        self.set_planning_interface(user_joint_names, planning_group, ee_link_name)

        # Time parameterization
        self.timestep = timestep
        self.joint_vel_limits = joint_vel_limits
        self.joint_acc_limits = joint_acc_limits

    # -------------------------------------------------------------------------- #
    # Planning interface
    # -------------------------------------------------------------------------- #
    def set_planning_interface(self, user_joint_names, planning_group, ee_link_name):
        if user_joint_names is None:
            user_joint_names = self.robot.active_joint_names
        assert len(user_joint_names) == len(
            self.robot.active_joint_names
        ), "user_joint_names ({}) should be as long as active joints ({}).".format(
            len(user_joint_names), len(self.robot.active_joint_names)
        )
        self._user_joint_names = user_joint_names

        support_joint_ids = self.robot.get_support_joint_ids(ee_link_name)
        support_group = [self.robot.joint_names[i] for i in support_joint_ids]
        if planning_group is None:
            planning_group = support_group

        self._planning_group = planning_group
        self._ee_link_name = ee_link_name

        self.name_to_idx_q = {}
        self.pin2user = []  # mapping from pinocchio qpos to user qpos
        self.mask_plan = np.zeros(self.robot.nq, dtype=bool)  # pinocchio order
        self.mask_support = np.zeros(self.robot.nq, dtype=bool)  # pinocchio order
        for name in user_joint_names:
            idx_q = self.robot.joint_idx_q(name)
            nq = self.robot.joint_nq(name)
            if nq > 1:
                logger.warning("%s has more than 1 DoF", name)
            self.name_to_idx_q[name] = (idx_q, nq)
            self.pin2user.extend(range(idx_q, idx_q + nq))
            if name in planning_group:
                self.mask_plan[idx_q : idx_q + nq] = 1
            if name in support_group:
                self.mask_support[idx_q : idx_q + nq] = 1
        self.pin2user = np.array(self.pin2user)
        self.user2pin = np.arange(self.robot.nq)[self.pin2user]
        # "inactive" means not being planned in fact
        self.mask_inactive = np.logical_not(
            np.logical_and(self.mask_plan, self.mask_support)
        )

    @property
    def user_joint_names(self):
        return self._user_joint_names

    @user_joint_names.setter
    def user_joint_names(self, user_joint_names):
        self.set_planning_interface(
            user_joint_names, self._planning_group, self._ee_link_name
        )

    @property
    def planning_group(self):
        return self._planning_group

    @planning_group.setter
    def planning_group(self, planning_group):
        self.set_planning_interface(
            self._user_joint_names, planning_group, self._ee_link_name
        )

    @property
    def dof(self):
        return len(self._planning_group)

    @property
    def ee_link_name(self):
        return self._ee_link_name

    @ee_link_name.setter
    def ee_link_name(self, ee_link_name):
        return self.set_planning_interface(
            self._user_joint_names, self._planning_group, ee_link_name
        )

    def get_ee_pose(self, qpos):
        qpos = np.array(qpos)[self.user2pin]
        frame_id = self.robot.frame_index(self._ee_link_name)
        # ee pose at base frame
        ee_pose = self.robot.framePlacement(qpos, frame_id)
        return np.array(self.robot._base_pose * ee_pose)

    @property
    def scene(self):
        """Get the planning scene."""
        # We reuse RobotWrapper here. Implement an individual PlanningScene if necessary.
        return self.robot

    # -------------------------------------------------------------------------- #
    # Algorithms
    # -------------------------------------------------------------------------- #
    def compute_CLIK(
        self,
        goal_pose,
        init_qpos,
        max_trials=1,
        check_collision=True,
        seed=None,
        **kwargs,
    ):
        """Closed-Loop Inverse Kinematics.

        Args:
            goal_pose: goal pose of end-effector. Refer to `to_SE3` for all supported formats.
            init_qpos: initial joint configuration for optimization.
            max_trials (int, optional): maximum numbers of initial (random) configurations to solve.
                Defaults to 1. If equals to 1, then only init_qpos is tried.
            check_collision (bool, optional): whether to check collision for IK solutions. Defaults to True.
            seed (int, optional): random seed.
            kwargs: hyperparameters for CLIK algorithm

        Returns:
            np.ndarray: [N, nq], IK solutions.
        """
        assert (
            len(init_qpos) == self.robot.nq
        ), "The length of init_qpos({}) is expected to be equal to nq({})".format(
            len(init_qpos), self.robot.nq
        )

        init_qpos = np.array(init_qpos)[self.user2pin]
        goal_pose = toSE3(goal_pose)
        _init_qpos = init_qpos
        results = []

        # NOTE(jigu): do not use pinocchio random configuration since I do not know how to fix random seed.
        state_space = JointStateSpace(*self.robot.joint_limits)
        rng = np.random.RandomState(seed)

        # Modify state space for joints not to plan
        state_space.low[self.mask_inactive] = init_qpos[self.mask_inactive]
        state_space.high[self.mask_inactive] = init_qpos[self.mask_inactive]

        for _ in range(max_trials):
            goal_qpos, ik_succ, ik_err = self.robot.compute_CLIK(
                goal_pose,
                self._ee_link_name,
                _init_qpos,
                qmask=self.mask_inactive,
                **kwargs,
            )

            if ik_succ:
                if check_collision and not self.robot.isCollisionFree(goal_qpos):
                    logger.debug("Find a solution of IK, but not collision-free")
                    ik_succ = False

                if not self.robot.within_joint_limits(goal_qpos):
                    logger.debug("Find a solution of IK, but out of joint limits")
                    ik_succ = False
            else:
                logger.debug("Fail to solve IK. The error is {}".format(ik_err))

            if ik_succ:
                logger.debug("Find a solution of IK")
                results.append(goal_qpos)

            # Try another configuration
            _init_qpos = state_space.sample_uniform(rng)

        results = np.array(results).reshape(-1, len(_init_qpos))
        results = results[:, self.pin2user]
        return results

    def plan_birrt(
        self,
        goal,
        start_qpos,
        ik_max_trials=20,
        rrt_range=0.1,
        rrt_max_iter=10000,
        start_qpos_range=0.0,
        start_qpos_max_trials=100,
        seed=None,
        goal_fmt="pose",
    ):
        """Plan a path by RRT-Connect (BiRRT).

        Args:
            goal: goal pose of end-effector or goal configuration.
            start_qpos (np.ndarray): start configuration
            rrt_range (float, optional): RRT range. Defaults to 0.1.
                It represents the maximum length of a motion to be added in the tree of motions.
            rrt_max_iter (int, optional): maximum number of trials to expand trees. Defaults to 10000.
            start_qpos_range (int, optional): The range around the provided @start_qpos,
                to sample a valid (collision-free) starting qpos. 0 disables sampling.
            start_qpos_max_trials (int): Number of attempts when sampling around @start_qpos.
                Has no effect if `start_qpos_range` is 0.
            seed: random seed.
            goal_fmt (str, optional): ["pose", "qpos"].

        Returns:
           dict:
            - status (str): success or other failure status
            - reason (str, optional): failure reason
            - position (np.ndarray, optional): [N, nq], found path.

        See also:
            - https://ompl.kavrakilab.org/classompl_1_1geometric_1_1RRTConnect.html
        """
        if goal_fmt == "qpos":
            goal_qpos = np.array(goal)
            assert goal_qpos.ndim <= 2, goal_qpos.shape
            goal_qpos = goal_qpos.reshape(-1, self.robot.nq)
        else:
            # TODO(jigu): add options for IK
            goal_qpos = self.compute_CLIK(
                goal,
                start_qpos,
                max_trials=ik_max_trials,
                check_collision=True,
                seed=seed,
            )
            if len(goal_qpos) == 0:
                logger.info("IK_FAILURE: Fail to find a feasible configuration by IK.")
                return {
                    "status": "ik_failure",
                    "reason": "Fail to find a feasible configuration by IK.",
                }

        # Map from user to pinocchio order
        goal_qpos = goal_qpos[:, self.user2pin]
        start_qpos = np.array(start_qpos)[self.user2pin]

        state_space = JointStateSpace(*self.robot.joint_limits)
        # state_space.set_state_validity_checker(lambda x: True)
        state_space.set_state_validity_checker(self.robot.isCollisionFree)
        goal_space = GoalStates(goal_qpos, state_space, 1e-3)

        # Modify state space for inactive joints
        state_space.low[self.mask_inactive] = start_qpos[self.mask_inactive]
        state_space.high[self.mask_inactive] = start_qpos[self.mask_inactive]

        rrt = RRTConnect(state_space)
        rrt.setup(
            [start_qpos],
            goal_space.goal,
            max_dist=rrt_range,
            max_iter=rrt_max_iter,
            start_state_range=start_qpos_range,
            start_state_max_trials=start_qpos_max_trials,
            seed=seed,
        )
        path = rrt.solve()

        if rrt.status != "success":
            logger.info("RRT_FAILURE: {}".format(rrt.status))
            result = dict(status="rrt_failure", reason=rrt.status)
            return result

        path = np.array(path)
        path = path[:, self.pin2user][:, self.mask_plan[self.pin2user]]
        result = dict(status="success")
        result["position"] = path

        if self.timestep is not None:
            result.update(
                parameterize_path(
                    path, self.joint_vel_limits, self.joint_acc_limits, self.timestep
                )
            )

        return result

    def plan_screw(
        self,
        goal_pose,
        start_qpos,
        qpos_step=0.1,
        goal_thresh=5e-3,
        screw_step=0.1,
        check_joint_limits=True,
    ):
        """Plan a path by screw motion.

        Args:
            goal_pose: goal pose of end-effector
            start_qpos: [nq], start configuration
            qpos_step: maximum norm of configuration per step
            goal_thresh: maximum norm of screw motion when a goal is reached.
            screw_step: maximum norm of screw motion per step.

        Returns:
            dict: same as `plan_by_birrt`

        See also:
            http://ras.papercept.net/images/temp/IROS/files/1984.pdf
        """
        start_qpos = np.array(start_qpos)[self.user2pin]

        frame_id = self.robot.frame_index(self._ee_link_name)
        # Target EE pose at base frame
        goal_pose = self.robot._base_pose.inverse() * toSE3(goal_pose)

        qpos = start_qpos
        path = [qpos]
        result = {}
        MAX_ITERS = 1000

        for _ in range(MAX_ITERS):
            # Current EE pose at base frame
            curr_pose = self.robot.framePlacement(qpos, frame_id)
            # Motion recorded in the spatial (base) frame
            delta_pose = goal_pose * curr_pose.inverse()
            # Screw motion (twist in a unit time)
            desired_screw = np.array(pin.log6(delta_pose))
            screw_norm = np.linalg.norm(desired_screw)

            # Reach the goal
            if screw_norm < goal_thresh:
                result["status"] = "success"
                break

            # Interpolate
            if screw_norm > screw_step:
                desired_screw = desired_screw * (screw_step / screw_norm)

            # NOTE(jigu): J * qv = v, so I assume J * (qv * dt) = v * dt
            # Solve desired joint velocities by IK
            J = self.robot.computeFrameJacobianWorld(qpos, frame_id)
            delta_qpos, residual, _, _ = np.linalg.lstsq(J, desired_screw, rcond=None)
            if residual > goal_thresh:
                result["status"] = "ik_failure"
                result["reason"] = f"The residual is {residual}"
                break

            # Update configuration
            delta_qnorm = np.linalg.norm(delta_qpos)
            if delta_qnorm > qpos_step:
                delta_qpos = delta_qpos * (qpos_step / delta_qnorm)
            qpos = qpos + delta_qpos

            # Check collision
            if self.robot.computeCollisions(qpos):
                result["status"] = "plan_failure"
                result["reason"] = "collision"
                break

            # Check joint limits
            if check_joint_limits:
                within_limits = self.robot.within_joint_limits(qpos, return_mask=True)
                if not np.all(within_limits):
                    logger.debug(
                        "within joint limits: {}".format(within_limits.tolist())
                    )
                    result["status"] = "plan_failure"
                    result["reason"] = "joint limits"
                    break

            # Add next configuration into path
            path.append(qpos)
        else:
            result["status"] = "plan_failure"
            result["reason"] = "timeout"

        path = np.array(path)
        path = path[:, self.pin2user][:, self.mask_plan[self.pin2user]]

        if result["status"] == "success":
            result["position"] = path
            if self.timestep is not None:
                result.update(
                    parameterize_path(
                        path,
                        self.joint_vel_limits,
                        self.joint_acc_limits,
                        self.timestep,
                    )
                )
        else:
            # Return partial solutions, which might be helpful for debugging
            result["position"] = path
        return result


try:
    import toppra as ta
    import toppra.algorithm as algo
    import toppra.constraint as constraint

    ta.setup_logging()
    logging.getLogger("toppra").propagate = False
except ImportError:
    logger.warn(
        "toppra is not installed for time parameterization (`pip install toppra`)."
    )


def parameterize_path(waypoints: np.ndarray, vlims, alims, timestep):
    # https://hungpham2511.github.io/toppra/auto_examples/plot_kinematics.html
    # computing the time-optimal path parametrization for robots subject to kinematic and dynamic constraints
    N, dof = waypoints.shape
    assert vlims is not None and alims is not None
    vlims = np.broadcast_to(vlims, dof)
    alims = np.broadcast_to(alims, dof)
    if N == 1:
        logger.warning("Only one waypoint. Skip time parameterization")
        return {}

    ss = np.linspace(0, 1, N)
    path = ta.SplineInterpolator(ss, waypoints)
    # If only one value is given, then the bound is [-value, value]
    pc_vel = constraint.JointVelocityConstraint(vlims)
    pc_acc = constraint.JointAccelerationConstraint(alims)

    instance = algo.TOPPRA([pc_vel, pc_acc], path, parametrizer="ParametrizeConstAccel")
    jnt_traj = instance.compute_trajectory()
    if jnt_traj is None:
        logger.warning("Fail to parameterize path.")
        return {}

    T = int(jnt_traj.duration / timestep)
    ts_sample = np.linspace(0, jnt_traj.duration, T)
    qs_sample = jnt_traj(ts_sample)
    qds_sample = jnt_traj(ts_sample, 1)
    qdds_sample = jnt_traj(ts_sample, 2)
    return dict(
        position=qs_sample,
        velocity=qds_sample,
        acceleration=qdds_sample,
        duration=jnt_traj.duration,
        time=ts_sample,
    )
