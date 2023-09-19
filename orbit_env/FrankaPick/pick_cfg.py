# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np

from omni.isaac.orbit.objects import RigidObjectCfg
from omni.isaac.orbit.robots.single_arm import SingleArmManipulatorCfg
from omni.isaac.orbit.utils import configclass
# from omni.isaac.orbit.actuators.group.actuator_group_cfg import ActuatorControlCfg
from omni.isaac.orbit.actuators.group import ActuatorGroupCfg
from omni.isaac.orbit.actuators.model import ImplicitActuatorCfg
from omni.isaac.orbit.actuators.group import ActuatorControlCfg, GripperActuatorGroupCfg
from omni.isaac.orbit.sensors.camera.camera_cfg import PinholeCameraCfg

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
                    "panda_joint2": -0.7850000262260437,
                    "panda_joint3": 0.0,
                    "panda_joint4": -2.3559999465942383,
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
                dof_pos_offset={"panda_joint5": 0.0, "panda_joint6": 1.57079632679, "panda_joint7": 0.7850000262260437},
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

camera_config = PinholeCameraCfg(
    sensor_tick=0.01,  # 例如每20毫秒一个传感器缓冲
    data_types=["rgb"],  # 例如启用颜色和深度数据
    width=160,
    height=120,
    projection_type="pinhole",  # 假设您使用的是针孔投影
    # semantic_types=["class"],  # 填写您需要的语义类型
    usd_params=PinholeCameraCfg.UsdCameraCfg(
        focal_length=24.0  # 假设您要设置的焦距为50.0mm
    )
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
        position_uniform_min = [0.5, -0.2, 0.2]  # position (x,y,z)
        position_uniform_max = [0.5, 0.2, 0.2]  # position (x,y,z)

    # initialize
    object_initial_pose: ObjectInitialPoseCfg = ObjectInitialPoseCfg()


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg:
        """Observations for policy group."""

        # global group settings
        enable_corruption: bool = False
        # observation terms
        # -- joint state
        arm_dof_pos = {"scale": 1.0}
        # arm_dof_pos_scaled = {"scale": 1.0}
        arm_dof_vel = {"scale": 0.5, "noise": {"name": "uniform", "min": -0.01, "max": 0.01}}
        # tool_dof_pos_scaled = {"scale": 1.0}
        # -- end effector state
        tool_positions = {"scale": 1.0}
        tool_orientations = {"scale": 1.0}
        # -- object state
        # object_positions = {"scale": 1.0}
        # object_orientations = {"scale": 1.0}
        # -- object tool relation
        # object_relative_tool_positions = {"scale": 1.0}
        # object_relative_tool_orientations = {"scale": 1.0}

        # -- previous action
        arm_actions = {"scale": 1.0}
        tool_actions = {"scale": 1.0}
        # camera
        camera_raw_output_normalized = {"scale": 1.0}

    # global observation settings
    return_dict_obs_in_group = False
    """Whether to return observations as dictionary or flattened vector within groups."""
    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    # -- robot-centric
    reaching_object_position_tanh = {"weight": 25, "sigma": 0.25}
    # reaching_object_position_tool_sites_tanh = {"weight": 25, "sigma": 0.1}
    # reaching_object_orientation_tanh = {"weight": 2.5, "sigma": 0.1}
    # penalizing_arm_dof_velocity_l2 = {"weight": 1e-4}
    # penalizing_tool_dof_velocity_l2 = {"weight": 1e-4}

    # -- action-centric
    penalizing_arm_action_rate_l2 = {"weight": 5}
    # penalizing_tool_action_l2 = {"weight": 5}
    # picking_object_success = {"weight": 100, "threshold": 0.25}
    reaching_object_success = {"weight": 100, "threshold": 0.06}
    # green_pixel_reward = {"weight": 1, "thresholds": [0.05, 0.25, 0.1]}
    # penalizing_tool_rotation = {"weight": 10}


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    episode_timeout = True  # reset when episode length ended
    object_falling = True  # reset when object falls off the table
    is_success = True  # reset when object is lifted


@configclass
class ControlCfg:
    """Processing of MDP actions."""

    # action space
    control_type = "default"  # "default", "inverse_kinematics"
    # decimation: Number of control action updates @ sim dt per policy dt
    decimation = 2


@configclass
class ActionCfg:
    action_low: np.ndarray = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0])
    action_high: np.ndarray = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
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

    # MDP settings
    randomization: RandomizationCfg = RandomizationCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Controller settings
    control: ControlCfg = ControlCfg()

    # camera
    camera: camera_config = camera_config

    # actions
    actionLimit: ActionCfg = ActionCfg()
