import argparse
import os

from omni.isaac.kit import SimulationApp

# add argparse arguments
parser = argparse.ArgumentParser("Welcome to Orbit: Omniverse Robotics Environments!")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
parser.add_argument("--gpu", action="store_true", default=False, help="Use GPU device for camera rendering output.")
parser.add_argument("--save", action="store_true", default=False, help="Save data or not")

args_cli = parser.parse_args()

# launch omniverse app
config = {"headless": args_cli.headless}
simulation_app = SimulationApp(config)


"""Rest everything follows."""
GRAPH_PATH = "/ActionGraph"
FRANKA_STAGE_PATH = "/World/Robot"
CAMERA_PRIM_PATH = f"{FRANKA_STAGE_PATH}/panda_hand/CameraSensor"
VIEWPORT_NAME = "viewport"

import torch
import cv2
import numpy as np

import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.core.utils import (  # noqa E402
    extensions
)
import omni.isaac.orbit.utils.kit as kit_utils
from omni.isaac.orbit.robots.config.franka import FRANKA_PANDA_ARM_WITH_PANDA_HAND_CFG
from omni.isaac.orbit.robots.config.universal_robots import UR10_CFG
from omni.isaac.orbit.robots.single_arm import SingleArmManipulator
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.orbit.sensors.camera import Camera, PinholeCameraCfg

import omni.graph.core as og  # noqa E402
import omni
"""
Main
"""


def main():
    """Spawns a single arm manipulator and applies random joint commands."""
    extensions.enable_extension("omni.isaac.ros2_bridge")
    # Load kit helper
    sim = SimulationContext(physics_dt=0.01, rendering_dt=0.01, backend="torch")
    # Set main camera
    set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    # Spawn things into stage
    # Ground-plane
    kit_utils.create_ground_plane("/World/defaultGroundPlane", z_position=0)
    # Lights-1
    prim_utils.create_prim(
        "/World/Light/GreySphere",
        "SphereLight",
        translation=(4.5, 3.5, 10.0),
        attributes={"radius": 2.5, "intensity": 600.0, "color": (0.75, 0.75, 0.75)},
    )
    # Lights-2
    prim_utils.create_prim(
        "/World/Light/WhiteSphere",
        "SphereLight",
        translation=(-4.5, 3.5, 10.0),
        attributes={"radius": 2.5, "intensity": 600.0, "color": (1.0, 1.0, 1.0)},
    )
    # Table
    table_usd_path = "../Props/TABLE.usd"
    prim_utils.create_prim("/World/Table", usd_path=table_usd_path, translation=(0.0, 0.0, 0.0))
    base_usd_path = "../Props/FRBASE.usd"
    prim_utils.create_prim("/World/Base", usd_path=base_usd_path, translation=(0.0, 0.0, 0.0))

    # Objects
    milk_usd_path = "../Props/Milk.usd"
    prim_utils.create_prim("/World/Milk", usd_path=milk_usd_path, translation=(0.22, -0.2, 0.6))
    yogurt_usd_path = "../Props/Yogurt.usd"
    prim_utils.create_prim("/World/Yogurt", usd_path=yogurt_usd_path, translation=(0.22, -0.1, 0.6))
    cheese_usd_path = "../Props/CreamCheese.usd"
    prim_utils.create_prim("/World/CreamCheese", usd_path=cheese_usd_path, translation=(0.22, 0.0, 0.6))
    butter_usd_path = "../Props/Butter.usd"
    prim_utils.create_prim("/World/Butter", usd_path=butter_usd_path, translation=(0.22, 0.1, 0.6))
    juice_usd_path = "../Props/OrangeJuice.usd"
    prim_utils.create_prim("/World/OrangeJuice", usd_path=juice_usd_path, translation=(0.22, 0.2, 0.6))
    parmesan_usd_path = "../Props/Parmesan.usd"
    prim_utils.create_prim("/World/Parmesan", usd_path=parmesan_usd_path, translation=(0.22, 0.0, 0.7))

    simulation_app.update()

    # ROS2 
    try:
        ros_domain_id = int(os.environ["ROS_DOMAIN_ID"])
        print("Using ROS_DOMAIN_ID: ", ros_domain_id)
    except ValueError:
        print("Invalid ROS_DOMAIN_ID integer value. Setting value to 0")
        ros_domain_id = 0
    except KeyError:
        print("ROS_DOMAIN_ID environment variable is not set. Setting value to 0")
        ros_domain_id = 0

    # Creating a action graph with ROS component nodes
    try:
        og.Controller.edit(
            {"graph_path": GRAPH_PATH, "evaluator_name": "execution"},
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnImpulseEvent", "omni.graph.action.OnImpulseEvent"),
                    ("ReadSimTime", "omni.isaac.core_nodes.IsaacReadSimulationTime"),
                    ("Context", "omni.isaac.ros2_bridge.ROS2Context"),
                    ("PublishJointState", "omni.isaac.ros2_bridge.ROS2PublishJointState"),
                    (
                        "SubscribeJointState",
                        "omni.isaac.ros2_bridge.ROS2SubscribeJointState",
                    ),
                    (
                        "ArticulationController",
                        "omni.isaac.core_nodes.IsaacArticulationController",
                    ),
                    ("PublishClock", "omni.isaac.ros2_bridge.ROS2PublishClock"),
                    ("OnTick", "omni.graph.action.OnTick"),
                    ("createViewport", "omni.isaac.core_nodes.IsaacCreateViewport"),
                    (
                        "getRenderProduct",
                        "omni.isaac.core_nodes.IsaacGetViewportRenderProduct",
                    ),
                    ("setCamera", "omni.isaac.core_nodes.IsaacSetCameraOnRenderProduct"),
                    ("cameraHelperRgb", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
                    ("cameraHelperInfo", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
                    ("cameraHelperDepth", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
                ],
                og.Controller.Keys.CONNECT: [
                    ("OnImpulseEvent.outputs:execOut", "PublishJointState.inputs:execIn"),
                    ("OnImpulseEvent.outputs:execOut", "SubscribeJointState.inputs:execIn"),
                    ("OnImpulseEvent.outputs:execOut", "PublishClock.inputs:execIn"),
                    (
                        "OnImpulseEvent.outputs:execOut",
                        "ArticulationController.inputs:execIn",
                    ),
                    ("Context.outputs:context", "PublishJointState.inputs:context"),
                    ("Context.outputs:context", "SubscribeJointState.inputs:context"),
                    ("Context.outputs:context", "PublishClock.inputs:context"),
                    (
                        "ReadSimTime.outputs:simulationTime",
                        "PublishJointState.inputs:timeStamp",
                    ),
                    ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                    (
                        "SubscribeJointState.outputs:jointNames",
                        "ArticulationController.inputs:jointNames",
                    ),
                    (
                        "SubscribeJointState.outputs:positionCommand",
                        "ArticulationController.inputs:positionCommand",
                    ),
                    (
                        "SubscribeJointState.outputs:velocityCommand",
                        "ArticulationController.inputs:velocityCommand",
                    ),
                    (
                        "SubscribeJointState.outputs:effortCommand",
                        "ArticulationController.inputs:effortCommand",
                    ),
                    ("OnTick.outputs:tick", "createViewport.inputs:execIn"),
                    ("createViewport.outputs:execOut", "getRenderProduct.inputs:execIn"),
                    ("createViewport.outputs:viewport", "getRenderProduct.inputs:viewport"),
                    ("getRenderProduct.outputs:execOut", "setCamera.inputs:execIn"),
                    (
                        "getRenderProduct.outputs:renderProductPath",
                        "setCamera.inputs:renderProductPath",
                    ),
                    ("setCamera.outputs:execOut", "cameraHelperRgb.inputs:execIn"),
                    ("setCamera.outputs:execOut", "cameraHelperInfo.inputs:execIn"),
                    ("setCamera.outputs:execOut", "cameraHelperDepth.inputs:execIn"),
                    ("Context.outputs:context", "cameraHelperRgb.inputs:context"),
                    ("Context.outputs:context", "cameraHelperInfo.inputs:context"),
                    ("Context.outputs:context", "cameraHelperDepth.inputs:context"),
                    (
                        "getRenderProduct.outputs:renderProductPath",
                        "cameraHelperRgb.inputs:renderProductPath",
                    ),
                    (
                        "getRenderProduct.outputs:renderProductPath",
                        "cameraHelperInfo.inputs:renderProductPath",
                    ),
                    (
                        "getRenderProduct.outputs:renderProductPath",
                        "cameraHelperDepth.inputs:renderProductPath",
                    ),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("Context.inputs:domain_id", ros_domain_id),
                    # Setting the /Franka target prim to Articulation Controller node
                    ("ArticulationController.inputs:usePath", True),
                    ("ArticulationController.inputs:robotPath", FRANKA_STAGE_PATH),
                    ("PublishJointState.inputs:topicName", "isaac_joint_states"),
                    ("SubscribeJointState.inputs:topicName", "isaac_joint_commands"),
                    ("createViewport.inputs:name", VIEWPORT_NAME),
                    ("createViewport.inputs:viewportId", 1),
                    ("cameraHelperRgb.inputs:frameId", "sim_camera"),
                    ("cameraHelperRgb.inputs:topicName", "rgb"),
                    ("cameraHelperRgb.inputs:type", "rgb"),
                    ("cameraHelperInfo.inputs:frameId", "sim_camera"),
                    ("cameraHelperInfo.inputs:topicName", "camera_info"),
                    ("cameraHelperInfo.inputs:type", "camera_info"),
                    ("cameraHelperDepth.inputs:frameId", "sim_camera"),
                    ("cameraHelperDepth.inputs:topicName", "depth"),
                    ("cameraHelperDepth.inputs:type", "depth"),
                ],
            },
        )
    except Exception as e:
        print(e)

    # Robots
    robot_cfg = FRANKA_PANDA_ARM_WITH_PANDA_HAND_CFG

    # -- Spawn robot
    robot = SingleArmManipulator(cfg=robot_cfg)
    robot.spawn("/World/Robot", translation=(-0.273, 0.0, 0.38))

    # Setup camera sensor
    camera_cfg = PinholeCameraCfg(
        sensor_tick=0,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane", "normals", "motion_vectors"],
        usd_params=PinholeCameraCfg.UsdCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
    )
    camera = Camera(cfg=camera_cfg, device="cuda" if args_cli.gpu else "cpu")
    camera.spawn("/World/Robot/panda_hand/CameraSensor")

    # Play the simulator
    sim.reset()
    # Acquire handles
    # Initialize handles
    robot.initialize("/World/Robot.*")
    # Reset states
    robot.reset_buffers()

    # Sensors
    camera.initialize()
    position = [0.06, 0.0, 0.0]
    orientation = [0.0, 0.0, 1.0, 0.0]
    camera.set_world_pose_ros(position, orientation)

    simulation_app.update()

    # need to initialize physics getting any articulation..etc
    sim.initialize_physics()
    sim.play()

    # Dock the second camera window
    viewport = omni.ui.Workspace.get_window("Viewport")
    rs_viewport = omni.ui.Workspace.get_window(VIEWPORT_NAME)
    rs_viewport.dock_in(viewport, omni.ui.DockPosition.RIGHT)


    # Now we are ready!
    print("[INFO]: Setup complete...")

    # dummy actions
    actions = torch.rand(robot.count, robot.num_actions, device=robot.device)
    has_gripper = robot.cfg.meta_info.tool_num_dof > 0

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    # episode counter
    sim_time = 0.0
    ep_step_count = 0

    for _ in range(14):
        sim.render()

    # Simulate physics
    while simulation_app.is_running():
        # If simulation is stopped, then exit.
        if sim.is_stopped():
            break
        # If simulation is paused, then skip.
        if not sim.is_playing():
            sim.step(render=True)
            continue
        # reset
        if ep_step_count % 1000 == 0:
            sim_time = 0.0
            ep_step_count = 0
            # reset dof state
            dof_pos, dof_vel = robot.get_default_dof_state()
            robot.set_dof_state(dof_pos, dof_vel)
            robot.reset_buffers()
            # reset command
            actions = torch.rand(robot.count, robot.num_actions, device=robot.device)
            
            # reset gripper
            if has_gripper:
                actions[:, -1] = -1
            print("[INFO]: Resetting robots state...")
        # change the gripper action
        if ep_step_count % 200 == 0 and has_gripper:
            # flip command for the gripper
            actions[:, -1] = -actions[:, -1]
            print(f"[INFO]: [Step {ep_step_count:03d}]: Flipping gripper command...")
        # apply action to the robot
        robot.apply_action(actions)
        # print("cur_pos:", robot.data.dof_pos)
        # print("vels:", robot.data.dof_vel)
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        ep_step_count += 1
        # note: to deal with timeline events such as stopping, we need to check if the simulation is playing
        if sim.is_playing():
            # update buffers
            robot.update_buffers(sim_dt)
            camera.update(dt=0.0)
        
        # 获取RGB和深度数据
        rgb_image = camera.data.output["rgb"]
        depth_image = camera.data.output["distance_to_image_plane"]

        # 转换为NumPy数组（如果它们还不是的话）
        if not isinstance(rgb_image, np.ndarray):
            rgb_image = rgb_image.cpu().numpy()
        if not isinstance(depth_image, np.ndarray):
            depth_image = depth_image.cpu().numpy()

        # 转换深度数据以更好地显示为灰度图像
        depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_image_normalized, cv2.COLORMAP_JET)

        # 使用cv2显示图像
        # cv2.imshow('RGB Image', rgb_image)
        # cv2.imshow('Depth Image', depth_colored)

        # 按'q'退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Tick the Publish/Subscribe JointState, Publish TF and Publish Clock nodes each frame
        og.Controller.set(
            og.Controller.attribute("/ActionGraph/OnImpulseEvent.state:enableImpulse"), True
        )
        
if __name__ == "__main__":
    # Run the main function
    main()
    # Close the simulator
    simulation_app.close()
