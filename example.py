import os
import time

import numpy as np
import pinocchio as pin

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

    # Initial end-effector pose
    init_ee_pose = planner.scene.framePlacement(init_qpos, "panda_hand_tcp")

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

    # # Visualize IK results
    # planner.scene.play(ik_results.T, dt=0.5)
    # time.sleep(WAIT_TIME)

    plan_result = planner.plan_birrt([p, q], init_qpos, seed=1024)
    q_traj = plan_result["position"]
    q_traj2 = np.tile(init_qpos[-2:], [len(q_traj), 1])
    q_traj = np.concatenate([q_traj, q_traj2], 1)

    # Visualize planned trajectory
    planner.scene.play(q_traj.T, dt=0.1)
    time.sleep(WAIT_TIME)

    # Attach box
    last_qpos = q_traj[-1]
    box2world = planner.scene.getGeometry("box").placement
    frame2world = planner.scene.framePlacement(last_qpos, "panda_hand_tcp")
    box2frame = frame2world.inverse() * box2world
    planner.scene.attachBox([0.04, 0.04, 0.12], box2frame, "panda_hand_tcp")
    planner.scene.disableCollision("box")

    # Planning group can be modified
    planner.planning_group = planner.user_joint_names
    # If planning group is changed, you need to change joint vel and acc limits.
    # Here, we just disable time parameterization
    planner.timestep = None
    plan_result = planner.plan_birrt(init_ee_pose, last_qpos, seed=1024)
    q_traj = plan_result["position"]

    # Visualize planned trajectory
    planner.scene.viz.reload(
        planner.scene.getGeometry("attached_box"), pin.GeometryType.VISUAL
    )
    planner.scene.play(q_traj.T, dt=0.1)
    time.sleep(WAIT_TIME)


if __name__ == "__main__":
    main()
