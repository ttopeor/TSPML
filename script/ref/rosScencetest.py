import os
from omni.isaac.kit import SimulationApp
CONFIG = {"renderer": "RayTracedLighting", "headless": False}
simulation_app = SimulationApp(CONFIG)


from omni.isaac.orbit.robots.config.franka import FRANKA_PANDA_ARM_WITH_PANDA_HAND_CFG # noqa E402
from omni.isaac.orbit.robots.single_arm import SingleArmManipulator # noqa E402


from omni.isaac.core import SimulationContext  # noqa E402
from omni.isaac.core.utils import extensions # noqa E402
from omni.isaac.core_nodes.scripts.utils import set_target_prims  # noqa E402

import omni.graph.core as og  # noqa E402

TSPML_DIR = "/home/robot/Desktop/workspace/TSPML"
FRANKA_DIR = f"{TSPML_DIR}/Props/Franka/franka_alt_fingers.usd"

# enable ROS2 bridge extension
extensions.enable_extension("omni.isaac.ros2_bridge")
simulation_context = SimulationContext(stage_units_in_meters=1.0)
# viewports.set_camera_view([0.8, 0.8, 0.4], [0.0, 0.0, 0.0])


robot = SingleArmManipulator(FRANKA_PANDA_ARM_WITH_PANDA_HAND_CFG)
robot.spawn("/panda")


try:
    ros_domain_id = int(os.environ["ROS_DOMAIN_ID"])
    print("Using ROS_DOMAIN_ID: ", ros_domain_id)
except ValueError:
    print("Invalid ROS_DOMAIN_ID integer value. Setting value to 0")
    ros_domain_id = 0
except KeyError:
    print("ROS_DOMAIN_ID environment variable is not set. Setting value to 0")
    ros_domain_id = 0

simulation_app.update()

og.Controller.edit(
    {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
    {
        og.Controller.Keys.CREATE_NODES: [
            ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
            ("PublishJointState", "omni.isaac.ros2_bridge.ROS2PublishJointState"),
            ("SubscribeJointState", "omni.isaac.ros2_bridge.ROS2SubscribeJointState"),
            ("ArticulationController", "omni.isaac.core_nodes.IsaacArticulationController"),
            ("ReadSimTime", "omni.isaac.core_nodes.IsaacReadSimulationTime"),
        ],
        og.Controller.Keys.CONNECT: [
            ("OnPlaybackTick.outputs:tick", "PublishJointState.inputs:execIn"),
            ("OnPlaybackTick.outputs:tick", "SubscribeJointState.inputs:execIn"),
            ("OnPlaybackTick.outputs:tick", "ArticulationController.inputs:execIn"),

            ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),

            ("SubscribeJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
            ("SubscribeJointState.outputs:positionCommand", "ArticulationController.inputs:positionCommand"),
            ("SubscribeJointState.outputs:velocityCommand", "ArticulationController.inputs:velocityCommand"),
            ("SubscribeJointState.outputs:effortCommand", "ArticulationController.inputs:effortCommand"),
        ],
        og.Controller.Keys.SET_VALUES: [
            # Providing path to /panda robot to Articulation Controller node
            # Providing the robot path is equivalent to setting the targetPrim in Articulation Controller node
            ("ArticulationController.inputs:usePath", True),
            ("ArticulationController.inputs:robotPath", "/panda"),
            #("PublishJointState.inputs:topicName", "isaac_joint_states")
            #("SubscribeJointState.inputs:topicName", "isaac_joint_commands")
        ],
    },
)

# Setting the /panda target prim to Publish JointState node
set_target_prims(primPath="/ActionGraph/PublishJointState", targetPrimPaths=["/panda"])

simulation_app.update()
simulation_context.initialize_physics()
simulation_context.play()

while simulation_app.is_running():
    # Run with a fixed step size
    simulation_context.step(render=True)

simulation_context.stop()
simulation_app.close()
