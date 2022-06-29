import os
import time

import numpy as np

from pymp import Planner, toSE3

WAIT_TIME = 10


def main():
    urdf = os.path.join(os.path.dirname(__file__), "data/panda.urdf")
    planner = Planner(urdf, [], ee_link_name="panda_hand_tcp")

    # Initial joint positions
    # init_qpos = np.zeros(9)
    init_qpos = np.array([0, 0.2, 0.0, -2.62, 0.0, 2.94, 0.785, 0.04, 0.04])

    planner.scene.addBox([0.04, 0.04, 0.12], toSE3([0.7, 0, 0.06]), name="box")
    planner.scene.addBox(
        [0.1, 0.4, 0.2], toSE3([0.55, 0, 0.1]), color=(0, 1, 1, 1), name="obstacle"
    )
    # planner.robot.addPointCloud(np.random.rand(100, 3), resolution=0.01, pose=np.eye(4))

    # Visualize initial qpos
    planner.scene.initMeshcatDisplay(None)
    planner.scene.display(init_qpos)
    time.sleep(WAIT_TIME)

    # Goal end-effector pose
    p = [0.7, 0, 0.1]
    q = [0, 1, 0, 0]

    # Compute IK
    ik_results = planner.compute_CLIK([p, q], init_qpos, max_trials=20, seed=0)
    print(len(ik_results))

    # Visualize IK results
    planner.robot.play(np.array(ik_results).T, dt=1)
    time.sleep(WAIT_TIME)

    rrt_result = planner.plan_rrt([p, q], init_qpos)
    planner.robot.play(np.array(rrt_result["position"]).T, dt=0.5)
    time.sleep(WAIT_TIME)


if __name__ == "__main__":
    main()
