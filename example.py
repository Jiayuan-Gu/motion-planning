"""The example to use pymp to plan a path for pick-and-place.
The visualization is based on Meshcat. 
The usage of Meshcat visualizer is based on https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/md_doc_b-examples_display_b-meshcat-viewer.html. 
"""

import os
import time

import numpy as np
import pinocchio as pin

from pymp import Planner, toSE3


def main():
    WAIT_TIME = 1

    # Create planner
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

    # Initialize environment
    planner.scene.addBox([0.04, 0.04, 0.12], toSE3([0.7, 0, 0.06]), name="box")
    planner.scene.addBox(
        [0.1, 0.4, 0.2], toSE3([0.55, 0, 0.1]), color=(0, 1, 1, 1), name="obstacle"
    )
    # planner.robot.addOctree(np.random.rand(1000, 3), resolution=0.01)

    # Initialize visualizer
    from pinocchio.visualize import MeshcatVisualizer

    try:
        viz = MeshcatVisualizer(
            planner.scene.model,
            collision_model=planner.scene.collision_model,
            visual_model=planner.scene.visual_model,
        )
        viz.initViewer(open=True)
        viz.loadViewerModel()
    except ImportError as err:
        print("Install Meshcat for visualization: `pip install meshcat`")
        raise err

    # Visualize initial qpos
    viz.display(init_qpos)
    time.sleep(WAIT_TIME)

    # Goal end-effector pose
    p = [0.7, 0, 0.1]
    q = [0, 1, 0, 0]  # wxyz

    # Compute IK
    ik_results = planner.compute_CLIK([p, q], init_qpos, max_trials=20, seed=0)
    print("# IK solutions:", len(ik_results))
    # print(ik_results[:, -2:])

    # Visualize IK results
    viz.play(ik_results.T, dt=0.5)
    time.sleep(WAIT_TIME)

    # Use RRT-connect to plan a path
    plan_result = planner.plan_birrt([p, q], init_qpos, seed=1024)
    q_traj = plan_result["position"]  # [N, nq]
    # Add gripper positions to trajectory
    q_traj = np.pad(q_traj, [(0, 0), (0, 2)], constant_values=init_qpos[-2:])

    # Visualize planned trajectory
    viz.play(q_traj.T, dt=0.1)
    time.sleep(WAIT_TIME)

    # Attach box
    last_qpos = q_traj[-1]
    box2world = planner.scene.getGeometry("box").placement
    frame2world = planner.scene.framePlacement(last_qpos, "panda_hand_tcp")
    box2frame = frame2world.inverse() * box2world
    planner.scene.attachBox([0.04, 0.04, 0.12], box2frame, "panda_hand_tcp")

    # Remove the previous box
    # planner.scene.disableCollision("box")
    viz.delete(planner.scene.getGeometry("box"), pin.GeometryType.VISUAL)
    planner.scene.removeGeometry("box")

    # Planning group can be modified
    planner.planning_group = planner.user_joint_names
    # If planning group is changed, you need to change joint vel and acc limits.
    # Here, we just disable time parameterization
    planner.timestep = None
    plan_result = planner.plan_birrt(init_ee_pose, last_qpos, seed=1024)
    q_traj = plan_result["position"]

    # Visualize planned trajectory
    viz.reload(planner.scene.getGeometry("attached_box"), pin.GeometryType.VISUAL)
    viz.play(q_traj.T, dt=0.1)
    time.sleep(WAIT_TIME)


if __name__ == "__main__":
    main()
