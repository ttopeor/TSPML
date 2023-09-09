# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import random

from omni.isaac.orbit.controllers.differential_inverse_kinematics import DifferentialInverseKinematicsCfg
from omni.isaac.orbit.objects import RigidObjectCfg
from omni.isaac.orbit.robots.single_arm import SingleArmManipulatorCfg
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.orbit.actuators.group.actuator_group_cfg import ActuatorControlCfg
from omni.isaac.orbit.actuators.group import ActuatorGroupCfg
from omni.isaac.orbit.actuators.model import ImplicitActuatorCfg
from omni.isaac.orbit.actuators.group import ActuatorControlCfg, GripperActuatorGroupCfg

from omni.isaac.orbit_envs.isaac_env_cfg import EnvCfg, IsaacEnvCfg, PhysxCfg, SimCfg, ViewerCfg

TSPML_DIR = "/home/robot/Desktop/workspace/TSPML"
FRANKA_DIR = f"{TSPML_DIR}/Props/Franka/franka_alt_fingers.usd"
TABLE_DIR = f"{TSPML_DIR}/Props/TABLE.usd"
BASE_DIR = f"{TSPML_DIR}/Props/FRBASE.usd"
# Objects
BUTTER_DIR = f"{TSPML_DIR}/Props/Butter.usd"
CREAMCHEESE_DIR = f"{TSPML_DIR}/Props/CreamCheese.usd"
MILK_DIR = f"{TSPML_DIR}/Props/Milk.usd"
ORANGE_DIR = f"{TSPML_DIR}/Props/OrangeJuice.usd"
PARMESAN_DIR = f"{TSPML_DIR}/Props/Parmesan.usd"
YOGURT_DIR = f"{TSPML_DIR}/Props/Yogurt.usd"
TEST_DIR = f"{TSPML_DIR}/Props/test.usd"
OBJ_LIST = [BUTTER_DIR, CREAMCHEESE_DIR, MILK_DIR, ORANGE_DIR, PARMESAN_DIR, YOGURT_DIR]
##
# Scene settings
##
FRANKA_PANDA_ARM_WITH_PANDA_HAND_CFG = SingleArmManipulatorCfg(
    meta_info=SingleArmManipulatorCfg.MetaInfoCfg(
        usd_path=FRANKA_DIR,
        arm_num_dof=7,
        tool_num_dof=2,
        tool_sites_names=["panda_leftfinger", "panda_rightfinger"],
    ),
    init_state=SingleArmManipulatorCfg.InitialStateCfg(
        dof_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -0.7850000262260437,
            "panda_joint3": 0.0,
            "panda_joint4": -2.3559999465942383,
            "panda_joint5": 0.0,
            "panda_joint6": 1.57079632679,
            "panda_joint7": 0.7850000262260437,
            "panda_finger_joint*": 0.04,
        },
        dof_vel={".*": 0.0},
    ),
    ee_info=SingleArmManipulatorCfg.EndEffectorFrameCfg(
        body_name="panda_hand", pos_offset=(0.0, 0.0, 0.1034), rot_offset=(1.0, 0.0, 0.0, 0.0)
    ),
    rigid_props=SingleArmManipulatorCfg.RigidBodyPropertiesCfg(
        max_depenetration_velocity=5.0,
        disable_gravity=True,
    ),
    collision_props=SingleArmManipulatorCfg.CollisionPropertiesCfg(
        contact_offset=0.005,
        rest_offset=0.0,
    ),
    articulation_props=SingleArmManipulatorCfg.ArticulationRootPropertiesCfg(
        enable_self_collisions=True,
    ),
    actuator_groups={
        "panda_shoulder": ActuatorGroupCfg(
            dof_names=["panda_joint[1-4]"],
            model_cfg=ImplicitActuatorCfg(velocity_limit=100.0, torque_limit=87.0),
            control_cfg=ActuatorControlCfg(
                command_types=["p_abs"],
                stiffness={".*": 1e5},
                # damping={".*": 40.0},
                dof_pos_offset={
                    "panda_joint1": 0.0,
                    "panda_joint2": 0.0,
                    "panda_joint3": 0.0,
                    "panda_joint4": 0.0,
                },
            ),
        ),
        "panda_forearm": ActuatorGroupCfg(
            dof_names=["panda_joint[5-7]"],
            model_cfg=ImplicitActuatorCfg(velocity_limit=100.0, torque_limit=12.0),
            control_cfg=ActuatorControlCfg(
                command_types=["p_abs"],
                stiffness={".*": 1e5},
                # damping={".*": 40.0},
                dof_pos_offset={"panda_joint5": 0.0, "panda_joint6": 0.0, "panda_joint7": 0.0},
            ),
        ),
        "panda_hand": GripperActuatorGroupCfg(
            dof_names=["panda_finger_joint[1-2]"],
            model_cfg=ImplicitActuatorCfg(velocity_limit=0.2, torque_limit=200),
            control_cfg=ActuatorControlCfg(command_types=["p_abs"], stiffness={".*": 1e5}, damping={".*": 1e3}),
            mimic_multiplier={"panda_finger_joint.*": 1.0},
            speed=0.1,
            open_dof_pos=0.04,
            close_dof_pos=0.0,
        ),
    },
)

@configclass
class TableCfg:
    """Properties for the table."""

    # note: we use instanceable asset since it consumes less memory
    usd_path = TABLE_DIR

@configclass
class BaseCfg:
    """Properties for the robot base."""

    # note: we use instanceable asset since it consumes less memory
    usd_path = BASE_DIR

@configclass
class ManipulationObjectCfg(RigidObjectCfg):
    """Properties for the object to manipulate in the scene."""

    meta_info = RigidObjectCfg.MetaInfoCfg(
        usd_path=MILK_DIR,
        # usd_path=random.choice(OBJ_LIST),
    )

    init_state = RigidObjectCfg.InitialStateCfg(
        pos=(0.5, 0.0, 0.3), rot=(1.0, 0.0, 0.0, 0.0), lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0)
    )
    rigid_props = RigidObjectCfg.RigidBodyPropertiesCfg(
        solver_position_iteration_count=16,
        solver_velocity_iteration_count=1,
        max_angular_velocity=1000.0,
        max_linear_velocity=1000.0,
        max_depenetration_velocity=5.0,
        disable_gravity=False,
    )
    physics_material = RigidObjectCfg.PhysicsMaterialCfg(
        static_friction=0.5, dynamic_friction=0.5, restitution=0.0, prim_path="/World/Materials/cubeMaterial"
    )


@configclass
class GoalMarkerCfg:
    """Properties for visualization marker."""

    # usd file to import
    usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd"
    # scale of the asset at import
    scale = [0.05, 0.05, 0.05]  # x,y,z


@configclass
class FrameMarkerCfg:
    """Properties for visualization marker."""

    # usd file to import
    usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd"
    # scale of the asset at import
    scale = [0.1, 0.1, 0.1]  # x,y,z


##
# MDP settings
##


@configclass
class RandomizationCfg:
    """Randomization of scene at reset."""

    @configclass
    class ObjectInitialPoseCfg:
        """Randomization of object initial pose."""

        # category
        position_cat: str = "uniform"  # randomize position: "default", "uniform"
        orientation_cat: str = "uniform"  # randomize position: "default", "uniform"
        # randomize position
        position_uniform_min = [0.4, -0.25, 0.3]  # position (x,y,z)
        position_uniform_max = [0.6, 0.25, 0.3]  # position (x,y,z)

    @configclass
    class ObjectDesiredPoseCfg:
        """Randomization of object desired pose."""

        # category
        position_cat: str = "default"  # randomize position: "default", "uniform"
        orientation_cat: str = "default"  # randomize position: "default", "uniform"
        # randomize position
        position_default = [0.5, 0.0, 0.5]  # position default (x,y,z)
        position_uniform_min = [0.4, -0.25, 0.25]  # position (x,y,z)
        position_uniform_max = [0.6, 0.25, 0.5]  # position (x,y,z)
        # randomize orientation
        orientation_default = [1.0, 0.0, 0.0, 0.0]  # orientation default

    # initialize
    object_initial_pose: ObjectInitialPoseCfg = ObjectInitialPoseCfg()
    object_desired_pose: ObjectDesiredPoseCfg = ObjectDesiredPoseCfg()


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg:
        """Observations for policy group."""

        # global group settings
        enable_corruption: bool = True
        # observation terms
        # -- joint state
        arm_dof_pos = {"scale": 1.0}
        # arm_dof_pos_scaled = {"scale": 1.0}
        # arm_dof_vel = {"scale": 0.5, "noise": {"name": "uniform", "min": -0.01, "max": 0.01}}
        tool_dof_pos_scaled = {"scale": 1.0}
        # -- end effector state
        tool_positions = {"scale": 1.0}
        tool_orientations = {"scale": 1.0}
        # -- object state
        # object_positions = {"scale": 1.0}
        # object_orientations = {"scale": 1.0}
        object_relative_tool_positions = {"scale": 1.0}
        # object_relative_tool_orientations = {"scale": 1.0}
        # -- object desired state
        object_desired_positions = {"scale": 1.0}
        # -- previous action
        arm_actions = {"scale": 1.0}
        tool_actions = {"scale": 1.0}

    # global observation settings
    return_dict_obs_in_group = False
    """Whether to return observations as dictionary or flattened vector within groups."""
    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- robot-centric
    # reaching_object_position_l2 = {"weight": 0.0}
    # reaching_object_position_exp = {"weight": 2.5, "sigma": 0.25}
    reaching_object_position_tanh = {"weight": 2.5, "sigma": 0.1}
    # penalizing_arm_dof_velocity_l2 = {"weight": 1e-5}
    # penalizing_tool_dof_velocity_l2 = {"weight": 1e-5}
    # penalizing_robot_dof_acceleration_l2 = {"weight": 1e-7}
    # -- action-centric
    penalizing_arm_action_rate_l2 = {"weight": 1e-2}
    # penalizing_tool_action_l2 = {"weight": 1e-2}
    # -- object-centric
    # tracking_object_position_exp = {"weight": 5.0, "sigma": 0.25, "threshold": 0.08}
    tracking_object_position_tanh = {"weight": 5.0, "sigma": 0.2, "threshold": 0.08}
    picking_object_success = {"weight": 3.5, "threshold": 0.08}


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    episode_timeout = True  # reset when episode length ended
    object_falling = True  # reset when object falls off the table
    is_success = False  # reset when object is lifted


@configclass
class ControlCfg:
    """Processing of MDP actions."""

    # action space
    control_type = "default"  # "default", "inverse_kinematics"
    # decimation: Number of control action updates @ sim dt per policy dt
    decimation = 2

    # configuration loaded when control_type == "inverse_kinematics"
    inverse_kinematics: DifferentialInverseKinematicsCfg = DifferentialInverseKinematicsCfg(
        command_type="pose_rel",
        ik_method="dls",
        position_command_scale=(0.1, 0.1, 0.1),
        rotation_command_scale=(0.1, 0.1, 0.1),
    )


@configclass
class CameraCfg:
    # cam parameters
    HorizontalApertureAttr = 20.955
    VerticalApertureAttr = 15.7
    FocalLengthAttr = 18.8
    FocusDistanceAttr = 400

    # path
    camera_prim_path = "/Robot/panda_hand/geometry/realsense/realsense_camera"
    viewport_name = "viewport"


@configclass
class Ros2Cfg:
    # path
    franka_stage_path = "/Robot"
    graph_path = "/ActionGraph"

##
# Environment configuration
##


@configclass
class PickEnvCfg(IsaacEnvCfg):

    # General Settings
    env: EnvCfg = EnvCfg(num_envs=4096, env_spacing=2.5, episode_length_s=5.0)
    viewer: ViewerCfg = ViewerCfg(debug_vis=True, eye=(7.5, 7.5, 7.5), lookat=(0.0, 0.0, 0.0))
    # Physics settings
    sim: SimCfg = SimCfg(
        dt=0.01,
        substeps=1,
        physx=PhysxCfg(
            gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
            gpu_total_aggregate_pairs_capacity=16 * 1024,
            friction_correlation_distance=0.00625,
            friction_offset_threshold=0.01,
            bounce_threshold_velocity=0.2,
        ),
    )

    # Scene Settings
    # -- robot
    robot: SingleArmManipulatorCfg = FRANKA_PANDA_ARM_WITH_PANDA_HAND_CFG
    # -- object
    object: ManipulationObjectCfg = ManipulationObjectCfg()
    # -- table
    table: TableCfg = TableCfg()
    # -- base
    base: BaseCfg = BaseCfg()
    # -- visualization marker
    goal_marker: GoalMarkerCfg = GoalMarkerCfg()
    frame_marker: FrameMarkerCfg = FrameMarkerCfg()

    # MDP settings
    randomization: RandomizationCfg = RandomizationCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Controller settings
    control: ControlCfg = ControlCfg()

    # camera
    camera: CameraCfg = CameraCfg()
    # ros2 setting
    ros2: Ros2Cfg = Ros2Cfg()
