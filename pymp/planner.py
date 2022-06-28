import logging
import os
from typing import Sequence

import numpy as np
import pinocchio

from pymp.path_planning import RRT, GoalStates, JointStateSpace, RRTConnect
from pymp.robot import RobotWrapper
from pymp.utils import xyz_wijk_to_SE3

logger = logging.getLogger("pymp.planner")


class Planner:
    def __init__(
        self,
        urdf: str,
        planning_group: Sequence[str],
        ee_link_name: str,
        srdf: str = None,
        use_convex=True,
    ):
        self.urdf = urdf
        if not srdf:
            srdf = urdf.replace(".urdf", ".srdf")
            logger.warn("No SRDF provided. Use SRDF at {}.".format(srdf))
        if not os.path.exists(srdf):
            # TODO(jigu): generate SRDF if not exists
            raise FileNotFoundError(srdf)
        self.srdf = srdf

        # Initialize Pinocchio model
        self.robot = RobotWrapper.loadFromURDF(urdf, use_convex=use_convex)
        self.robot.initCollisionPairs()
        self.robot.removeCollisionPairsFromSRDF(srdf)

        self.planning_group = planning_group
        self.ee_link_name = ee_link_name

        # Options
        self.default_pose_fmt = "xyzwijk"

    @staticmethod
    def toSE3(x, fmt="xyzwijk"):
        if fmt == "xyzwijk":
            if len(x) == 2:  # x = (p, q)
                assert isinstance(x, (tuple, list)), type(x)
                pose = xyz_wijk_to_SE3(*x)
            else:
                pose = xyz_wijk_to_SE3(x[0:3], x[3:])
        elif fmt == "T":
            pose = pinocchio.SE3(np.array(x))
        elif fmt == "sapien":
            pose = xyz_wijk_to_SE3(x.p, x.q)
        else:
            raise NotImplementedError(fmt)
        return pose

    def compute_CLIK(
        self,
        goal_pose,
        init_qpos,
        max_trials=1,
        pose_fmt=None,
        check_collision=True,
        seed=None,
    ):
        goal_pose = self.toSE3(goal_pose, pose_fmt or self.default_pose_fmt)
        _init_qpos = init_qpos
        results = []

        # NOTE(jigu): do not use pinocchio random configuration since I do not know how to fix random seed.
        state_space = JointStateSpace(*self.robot.joint_limits)
        rng = np.random.RandomState(seed)

        for _ in range(max_trials):
            # TODO(jigu): add options for IK parameters
            goal_qpos, ik_succ, ik_err = self.robot.compute_CLIK(
                goal_pose, self.ee_link_name, _init_qpos
            )

            # TODO(jigu): check joint limits

            if ik_succ:
                if check_collision and not self.robot.isCollisionFree(goal_qpos):
                    logger.debug("Find a solution of IK, but not collision-free")
                    ik_succ = False
            else:
                logger.debug("Fail to solve IK. The error is {}".format(ik_err))

            if ik_succ:
                results.append(goal_qpos)

            # Try another configuration
            _init_qpos = state_space.sample_uniform(rng)

        return results

    def plan_rrt(self, goal, start_qpos, goal_fmt=None, seed=None):
        if goal_fmt == "qpos":
            goal_qpos = [goal]
        else:
            goal_qpos = self.compute_CLIK(
                goal,
                start_qpos,
                max_trials=20,
                pose_fmt=goal_fmt,
                check_collision=True,
                seed=seed,
            )
            if len(goal_qpos) == 0:
                return {
                    "status": "IK_FAILURE",
                    "reason": "Fail to find a feasible configuration by IK.",
                }

        state_space = JointStateSpace(*self.robot.joint_limits)
        # state_space.set_state_validity_checker(lambda x: True)
        state_space.set_state_validity_checker(self.robot.isCollisionFree)
        goal_space = GoalStates(goal_qpos, state_space, 1e-3)
        
        # rrt = RRT(state_space, goal_bias=0.05)
        # rrt.setup([start_qpos], goal_space.sample, goal_space.is_satisfied, 0.1, 100000)
        rrt = RRTConnect(state_space)
        rrt.setup([start_qpos], goal_space.goal, 0.1, 100000)
        rrt_result = rrt.solve()

        result = dict(position=np.array(rrt_result), status=rrt.status)
        print("RRT status", rrt.status)
        return result
