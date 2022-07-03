import os
import time

import numpy as np

from pymp import Planner, logger, toSE3

WAIT_TIME = 1
logger.setLevel("DEBUG")


def main():
    urdf = os.path.join(os.path.dirname(__file__), "data/panda.urdf")
    planner = Planner(
        urdf,
        None,
        ee_link_name="panda_hand_tcp",
        timestep=0.1,
        joint_vel_limits=1,
        joint_acc_limits=1,
    )

    # Initial joint positions
    # init_qpos = np.zeros(9)
    init_qpos = np.array([0, 0.2, 0.0, -2.62, 0.0, 2.94, 0.785, 0.04, 0.04])

    planner.scene.addBox([0.04, 0.04, 0.12], toSE3([0.7, 0, 0.06]), name="box")
    planner.scene.addBox(
        [0.1, 0.4, 0.2], toSE3([0.55, 0, 0.1]), color=(0, 1, 1, 1), name="obstacle"
    )
    # planner.robot.addOctree(np.random.rand(1000, 3), resolution=0.01)

    # Visualize initial qpos
    planner.scene.initMeshcatDisplay(None)
    planner.scene.display(init_qpos)
    time.sleep(WAIT_TIME)

    # Goal end-effector pose
    p = [0.7, 0, 0.1]
    q = [0, 1, 0, 0]

    # Compute IK
    ik_results = planner.compute_CLIK([p, q], init_qpos, max_trials=20, seed=0)
    print("# IK solutions:", len(ik_results))
    # print(ik_results[:, -2:])

    # Visualize IK results
    planner.robot.play(ik_results.T, dt=0.5)
    time.sleep(WAIT_TIME)

    plan_result = planner.plan_birrt([p, q], init_qpos, seed=1024)
    q_traj = plan_result["position"]
    q_traj2 = np.tile(init_qpos[-2:], [len(q_traj), 1])
    q_traj = np.concatenate([q_traj, q_traj2], 1)
    planner.robot.play(q_traj.T, dt=0.1)
    time.sleep(WAIT_TIME)


if __name__ == "__main__":
    main()
