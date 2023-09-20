# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gym.spaces
import math
import torch
from typing import List
import numpy as np
import re
import time

import omni.isaac.core.utils.prims as prim_utils

import omni.isaac.orbit.utils.kit as kit_utils
from omni.isaac.orbit.objects import RigidObject
from omni.isaac.orbit.robots.single_arm import SingleArmManipulator
from omni.isaac.orbit.sensors.camera import Camera
from omni.isaac.orbit.utils.dict import class_to_dict
from omni.isaac.orbit.utils.math import quat_inv, quat_mul, random_orientation, sample_uniform, scale_transform
from omni.isaac.orbit.utils.mdp import ObservationManager, RewardManager

from omni.isaac.orbit_envs.isaac_env import IsaacEnv, VecEnvIndices, VecEnvObs

from .pick_cfg import PickEnvCfg, RandomizationCfg
import omni.isaac.orbit.utils.array as array_utils
from scipy.spatial.transform import Rotation as R


class PickEnv(IsaacEnv):
    """Environment for pick an object off a table with a single-arm manipulator."""

    def __init__(self, cfg: PickEnvCfg = None, **kwargs):
        # copy configuration
        self.cfg = cfg
        # create classes (these are called by the function :meth:`_design_scene`)
        self.robot = SingleArmManipulator(cfg=self.cfg.robot)
        self.camera_prim = Camera(cfg=self.cfg.camera, device='cuda')
        self.object = RigidObject(cfg=self.cfg.object)
        
        self.cameras = []
        self.success_counters = None
        # initialize the base class to setup the scene.
        super().__init__(self.cfg, **kwargs)

        self.cameras = [Camera(cfg=self.cfg.camera, device='cuda') for _ in range(self.num_envs)]
        self.success_counters = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        # parse the configuration for information
        self._process_cfg()
        # initialize views for the cloned scenes
        self._initialize_views()

        # prepare the observation manager
        self._observation_manager = PickObservationManager(class_to_dict(self.cfg.observations), self, self.device)
        # prepare the reward manager
        self._reward_manager = PickRewardManager(
            class_to_dict(self.cfg.rewards), self, self.num_envs, self.dt, self.device
        )
        # print information about MDP
        print("[INFO] Observation Manager:", self._observation_manager)
        print("[INFO] Reward Manager: ", self._reward_manager)

        # compute the observation space: arm joint state + ee-position + goal-position + actions
        num_obs = self._observation_manager.group_obs_dim["policy"][0]
        self.observation_space = gym.spaces.Box(low=-math.inf, high=math.inf, shape=(num_obs,))
        # compute the action space
        action_low = self.cfg.actionLimit.action_low
        action_high = self.cfg.actionLimit.action_high
        # action_low[-1] = 1.0
        # action_high[-1] = 1.0
        self.action_space = gym.spaces.Box(low=action_low, high=action_high, shape=(self.num_actions,))
        print("[INFO]: Completed setting up the environment...")

        # Take an initial step to initialize the scene.
        self.sim.step()
        # -- fill up buffers
        self.object.update_buffers(self.dt)
        self.robot.update_buffers(self.dt)
        for camera in self.cameras:
            camera.update(self.dt)

    """
    Implementation specifics.
    """

    def _design_scene(self) -> List[str]:
        # ground plane
        kit_utils.create_ground_plane("/World/defaultGroundPlane", z_position=-0.37926)
        # table
        prim_utils.create_prim(
            self.template_env_ns + "/Table",
            usd_path=self.cfg.table.usd_path,
            position=np.array([0.27379, 0, -0.37926]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0])
        )
        # base
        prim_utils.create_prim(
            self.template_env_ns + "/Base",
            usd_path=self.cfg.base.usd_path,
            position=np.array([0.27379, 0, -0.37926]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0])
        )
        # robot
        self.robot.spawn(self.template_env_ns + "/Robot")
        # object
        self.object.spawn(self.template_env_ns + "/Object")
        # camera
        self.camera_prim.spawn(
            self.template_env_ns + "/Robot/panda_hand/geometry",
            translation=(0.05, 0.0, -0.88),  
            orientation=(0, 0.70711, 0.70711, 0.0)
        )
        # return list of global prims
        return ["/World/defaultGroundPlane"]

    def _reset_idx(self, env_ids: VecEnvIndices):
        # randomize the MDP
        # -- robot DOF state
        dof_pos, dof_vel = self.robot.get_default_dof_state(env_ids=env_ids)
        self.robot.set_dof_state(dof_pos, dof_vel, env_ids=env_ids)
        # -- object pose
        self._randomize_object_initial_pose(env_ids=env_ids, cfg=self.cfg.randomization.object_initial_pose)
        # -- reset Camera
        for camera in self.cameras:
            camera.reset()
        # -- Reward logging
        # fill extras with episode information
        self.extras["episode"] = dict()
        # reset
        # -- rewards manager: fills the sums for terminated episodes
        self._reward_manager.reset_idx(env_ids, self.extras["episode"])
        # -- obs manager
        self._observation_manager.reset_idx(env_ids)
        # -- reset history
        self.previous_actions[env_ids] = 0
        # -- MDP reset
        self.reset_buf[env_ids] = 0
        self.episode_length_buf[env_ids] = 0

    def _step_impl(self, actions: torch.Tensor):
        # pre-step: set actions into buffer
        self.actions = actions.clone().to(device=self.device)
        self.robot_actions[:] = self.actions

        # perform physics stepping
        for _ in range(self.cfg.control.decimation):
            # set actions into buffers
            self.robot.apply_action(self.robot_actions)
            # simulate
            self.sim.step(render=self.enable_render)
            # check that simulation is playing
            if self.sim.is_stopped():
                return
        # post-step:
        # -- compute common buffers
        self.robot.update_buffers(self.dt)
        self.object.update_buffers(self.dt)
        for camera in self.cameras:
            camera.update(self.dt)

        # -- compute MDP signals
        # reward
        self.reward_buf = self._reward_manager.compute()
        # terminations
        self._check_termination()
        # -- store history
        self.previous_actions = self.actions.clone()

        # -- add information to extra if timeout occurred due to episode length
        # Note: this is used by algorithms like PPO where time-outs are handled differently
        self.extras["time_outs"] = self.episode_length_buf >= self.max_episode_length
        # -- add information to extra if task completed
        left_finger_pos = self.robot.data.tool_sites_state_w[:, 0, :3] - self.envs_positions
        right_finger_pos = self.robot.data.tool_sites_state_w[:, 1, :3] - self.envs_positions
        ee_center_pos = (left_finger_pos + right_finger_pos) / 2
        object_pos = self.object.data.root_pos_w - self.envs_positions
        distance = torch.norm(ee_center_pos - object_pos, dim=1)
        self.extras["is_success"] = torch.where(distance < self.cfg.rewards.reaching_object_success["threshold"], 1.0, self.reset_buf)

    def _get_observations(self) -> VecEnvObs:
        # compute observations
        return self._observation_manager.compute()

    """
    Helper functions - Scene handling.
    """

    def _process_cfg(self) -> None:
        """Post processing of configuration parameters."""
        # compute constants for environment
        self.dt = self.cfg.control.decimation * self.physics_dt  # control-dt
        self.max_episode_length = math.ceil(self.cfg.env.episode_length_s / self.dt)

        # convert configuration parameters to torchee
        # randomization
        # -- initial pose
        config = self.cfg.randomization.object_initial_pose
        for attr in ["position_uniform_min", "position_uniform_max"]:
            setattr(config, attr, torch.tensor(getattr(config, attr), device=self.device, requires_grad=False))

    def _initialize_views(self) -> None:
        """Creates views and extract useful quantities from them."""
        # play the simulator to activate physics handles
        # note: this activates the physics simulation view that exposes TensorAPIs
        self.sim.reset()

        # define views over instances
        self.robot.initialize(self.env_ns + "/.*/Robot")
        self.object.initialize(self.env_ns + "/.*/Object")
        for i, camera in enumerate(self.cameras):
            camera.initialize(f"{self.env_ns}/env_{i}/Robot/panda_hand/geometry/Camera")

            
        # create controller
        self.num_actions = self.robot.num_actions

        # history
        self.actions = torch.zeros((self.num_envs, self.robot.num_actions), device=self.device)
        self.previous_actions = torch.zeros((self.num_envs, self.robot.num_actions), device=self.device)
        self.previous_camera_tensor = torch.zeros((self.num_envs, self.cfg.camera.width*self.cfg.camera.height*3), device=self.device)
        # robot joint actions
        # self.robot_actions = torch.zeros((self.num_envs, self.robot.num_actions), device=self.device)
        dof_pos, _ = self.robot.get_default_dof_state()
        dof_pos[0, -2] = 1.0 if dof_pos[0, -2] > 0.035 else -1.0
        dof_pos = dof_pos[:, :-1]
        self.robot_actions = dof_pos
        # buffers
        self.object_root_pose_ee = torch.zeros((self.num_envs, 7), device=self.device)
        # time-step = 0
        self.object_init_pose_w = torch.zeros((self.num_envs, 7), device=self.device)

        self.robot_init_ee_state_w = torch.tensor([[ 3.0698e-01, -3.0659e-08,  4.8685e-01,  2.0923e-07,  1.0000e+00,
          1.9913e-04, -1.0204e-04,  1.4948e-07,  2.2504e-08,  1.7103e-07,
          1.9246e-07, -1.5114e-06,  1.0964e-05]], device='cuda:0')

    """
    Helper functions - MDP.
    """

    def _check_termination(self) -> None:
        # access buffers from simulator
        object_pos = self.object.data.root_pos_w - self.envs_positions
        # extract values from buffer
        self.reset_buf[:] = 0
        # compute resets
        # -- when task is successful
        if self.cfg.terminations.is_success:
            left_finger_pos = self.robot.data.tool_sites_state_w[:, 0, :3] - self.envs_positions
            right_finger_pos = self.robot.data.tool_sites_state_w[:, 1, :3] - self.envs_positions
            ee_center_pos = (left_finger_pos + right_finger_pos) / 2
            distance = torch.norm(ee_center_pos - object_pos, dim=1)
            # Check if the distance is below the threshold for each environment
            is_success = distance < self.cfg.rewards.reaching_object_success["threshold"]
            # Increment counters where success is observed
            self.success_counters = torch.where(is_success, self.success_counters + 1, torch.zeros_like(self.success_counters))
  
            # Check where the success counter has reached 30
            termination_indices = self.success_counters >= 30
            # Update the reset buffer for environments that reached 100 consecutive successes
            self.reset_buf = torch.where(termination_indices, torch.ones_like(self.reset_buf), self.reset_buf)
            # Reset the counters for terminated environments
            self.success_counters = torch.where(termination_indices, torch.zeros_like(self.success_counters), self.success_counters)
        # -- object fell off the table (table at height: 0.0 m)
        if self.cfg.terminations.object_falling:
            self.reset_buf = torch.where(object_pos[:, 2] < -0.05, 1, self.reset_buf)
        # -- episode length
        if self.cfg.terminations.episode_timeout:
            self.reset_buf = torch.where(self.episode_length_buf >= self.max_episode_length, 1, self.reset_buf)

    def _randomize_object_initial_pose(self, env_ids: torch.Tensor, cfg: RandomizationCfg.ObjectInitialPoseCfg):
        """Randomize the initial pose of the object."""
        # get the default root state
        root_state = self.object.get_default_root_state(env_ids)
        # -- object root position
        if cfg.position_cat == "default":
            pass
        elif cfg.position_cat == "uniform":
            # sample uniformly from box
            # note: this should be within in the workspace of the robot
            root_state[:, 0:3] = sample_uniform(
                cfg.position_uniform_min, cfg.position_uniform_max, (len(env_ids), 3), device=self.device
            )
        else:
            raise ValueError(f"Invalid category for randomizing the object positions '{cfg.position_cat}'.")
        # -- object root orientation
        if cfg.orientation_cat == "default":
            pass
        elif cfg.orientation_cat == "uniform":
            # sample uniformly in SO(3)
            root_state[:, 3:7] = random_orientation(len(env_ids), self.device)
        else:
            raise ValueError(f"Invalid category for randomizing the object orientation '{cfg.orientation_cat}'.")
        # transform command from local env to world
        root_state[:, 0:3] += self.envs_positions[env_ids]
        # update object init pose
        self.object_init_pose_w[env_ids] = root_state[:, 0:7]
        # set the root state
        self.object.set_root_state(root_state, env_ids=env_ids)


class PickObservationManager(ObservationManager):
    """Reward manager for single-arm reaching environment."""

    def arm_dof_pos(self, env: PickEnv):
        """DOF positions for the arm."""
        return env.robot.data.arm_dof_pos

    def arm_dof_pos_scaled(self, env: PickEnv):
        """DOF positions for the arm normalized to its max and min ranges."""
        return scale_transform(
            env.robot.data.arm_dof_pos,
            env.robot.data.soft_dof_pos_limits[:, : env.robot.arm_num_dof, 0],
            env.robot.data.soft_dof_pos_limits[:, : env.robot.arm_num_dof, 1],
        )

    def arm_dof_vel(self, env: PickEnv):
        """DOF velocity of the arm."""
        return env.robot.data.arm_dof_vel

    def tool_dof_pos_scaled(self, env: PickEnv):
        """DOF positions of the tool normalized to its max and min ranges."""
        return scale_transform(
            env.robot.data.tool_dof_pos,
            env.robot.data.soft_dof_pos_limits[:, env.robot.arm_num_dof :, 0],
            env.robot.data.soft_dof_pos_limits[:, env.robot.arm_num_dof :, 1],
        )

    def tool_positions(self, env: PickEnv):
        """Current end-effector position of the arm."""
        return env.robot.data.ee_state_w[:, :3] - env.envs_positions

    def tool_orientations(self, env: PickEnv):
        """Current end-effector orientation of the arm."""
        # make the first element positive
        quat_w = env.robot.data.ee_state_w[:, 3:7]
        quat_w[quat_w[:, 0] < 0] *= -1
        return quat_w

    def object_positions(self, env: PickEnv):
        """Current object position."""
        return env.object.data.root_pos_w - env.envs_positions

    def object_orientations(self, env: PickEnv):
        """Current object orientation."""
        # make the first element positive
        quat_w = env.object.data.root_quat_w
        quat_w[quat_w[:, 0] < 0] *= -1
        return quat_w

    def object_relative_tool_positions(self, env: PickEnv):
        """Current object position w.r.t. end-effector frame."""
        return env.object.data.root_pos_w - env.robot.data.ee_state_w[:, :3]

    def object_relative_tool_orientations(self, env: PickEnv):
        """Current object orientation w.r.t. end-effector frame."""
        # compute the relative orientation
        quat_ee = quat_mul(quat_inv(env.robot.data.ee_state_w[:, 3:7]), env.object.data.root_quat_w)
        # make the first element positive
        quat_ee[quat_ee[:, 0] < 0] *= -1
        return quat_ee

    def arm_actions(self, env: PickEnv):
        """Last arm actions provided to env."""
        return env.actions[:, :-1]

    def tool_actions(self, env: PickEnv):
        """Last tool actions provided to env."""
        return env.actions[:, -1].unsqueeze(1)

    def tool_actions_bool(self, env: PickEnv):
        """Last tool actions transformed to a boolean command."""
        return torch.sign(env.actions[:, -1]).unsqueeze(1)

    def camera_raw_output_normalized(self, env: PickEnv):
        all_cameras_data = [None] * len(env.cameras)  # Initialize a list with None entries for each camera

        for idx, camera in enumerate(env.cameras):
            rgb_data = camera.data.output["rgb"]

            if rgb_data is not None:
                converted_tensor = array_utils.convert_to_torch(rgb_data, device=env.device)
                rgb_only_tensor = converted_tensor[:, :, :3]
                
                # Normalize the data to [0, 1] range
                rgb_only_tensor = rgb_only_tensor / 255.0
                
                rgb_only_tensor = rgb_only_tensor.contiguous().view(1, -1)  # shape it as [1, width*height*3]
                all_cameras_data[idx] = rgb_only_tensor

        # Use data from previous_camera_tensor for cameras that have no new data
        for idx, data in enumerate(all_cameras_data):
            if data is None:
                all_cameras_data[idx] = env.previous_camera_tensor[idx].unsqueeze(0)  # Add the previous data for this camera

        # Now, stack all tensors along the first dimension
        stacked_tensor = torch.cat(all_cameras_data, dim=0)
        env.previous_camera_tensor = stacked_tensor.clone()
        return stacked_tensor


class PickRewardManager(RewardManager):

    def reaching_object_position_tool_sites_tanh(self, env: PickEnv, sigma: float):
        """Penalize tool sites tracking position error using tanh-kernel."""
        # distance of end-effector to the object: (num_envs,)
        ee_distance = torch.norm(env.robot.data.ee_state_w[:, 0:3] - env.object.data.root_pos_w, dim=1)
        # distance of the tool sites to the object: (num_envs, num_tool_sites)
        object_root_pos = env.object.data.root_pos_w.unsqueeze(1)  # (num_envs, 1, 3)
        tool_sites_distance = torch.norm(env.robot.data.tool_sites_state_w[:, :, :3] - object_root_pos, dim=-1)
        # average distance of the tool sites to the object: (num_envs,)
        # note: we add the ee distance to the average to make sure that the ee is always closer to the object
        num_tool_sites = tool_sites_distance.shape[1]
        average_distance = (ee_distance + torch.sum(tool_sites_distance, dim=1)) / (num_tool_sites + 1)

        return 1 - torch.tanh(average_distance / sigma)

    def reaching_object_position_tanh(self, env: PickEnv, sigma: float):
        """Penalize tool sites tracking position error using tanh-kernel."""
        # 提取左右两个finger的位置
        left_finger_pos = env.robot.data.tool_sites_state_w[:, 0, :3] - env.envs_positions
        right_finger_pos = env.robot.data.tool_sites_state_w[:, 1, :3] - env.envs_positions
        # 计算EE center的位置（两个finger位置的中点）
        ee_center_pos = (left_finger_pos + right_finger_pos) / 2
        # 提取object位置
        object_pos = env.object.data.root_pos_w - env.envs_positions
        # 计算EE center到object的距离
        distance = torch.norm(ee_center_pos - object_pos, dim=1)
        # 根据distance和sigma计算reward
        reward = 1 - torch.tanh(distance / sigma)

        return reward

    def reaching_object_orientation_tanh(self, env: PickEnv, sigma: float):
        """Penalize tool sites tracking orientation error using tanh-kernel."""

        # Extract end-effector's orientation (quaternion): (num_envs, 4)
        ee_quat = env.robot.data.ee_state_w[:, 3:7]

        # Extract object's orientation (quaternion): (num_envs, 4)
        obj_quat = env.object.data.root_quat_w

        # Calculate the angle difference between two quaternions
        dot_product = torch.sum(ee_quat * obj_quat, dim=1)
        angle_difference = 2 * torch.acos(torch.clamp(dot_product, -1.0, 1.0))

        # We use tanh-kernel to compute the orientation difference
        return 1 - torch.tanh(angle_difference / sigma)

    # Penalties
    def penalizing_arm_dof_velocity_l2(self, env: PickEnv):
        """Penalize large movements of the robot arm."""
        return -torch.sum(torch.square(env.robot.data.arm_dof_vel), dim=1)

    def penalizing_tool_dof_velocity_l2(self, env: PickEnv):
        """Penalize large movements of the robot tool."""
        return -torch.sum(torch.square(env.robot.data.tool_dof_vel), dim=1)

    def penalizing_arm_action_rate_l2(self, env: PickEnv):
        """Penalize large variations in action commands besides tool."""
        return -torch.sum(torch.square(env.actions[:, :-1] - env.previous_actions[:, :-1]), dim=1)

    def penalizing_tool_action_l2(self, env: PickEnv):
        """Penalize large values in action commands for the tool."""
        return -torch.square(env.actions[:, -1])

    def penalizing_tool_rotation(self, env: PickEnv):
        # Extract quaternion from the states
        current_quat = env.robot.data.ee_state_w[:, 3:7].cpu().numpy()
        init_quat = env.robot_init_ee_state_w[:, 3:7].cpu().numpy()

        # Convert quaternions to euler angles (roll, pitch, yaw format)
        current_euler = R.from_quat(current_quat).as_euler('zyx', degrees=True)
        init_euler = R.from_quat(init_quat).as_euler('zyx', degrees=True)

        # Set yaw (z-rotation) to 0
        current_euler[:, 2] = 0
        init_euler[:, 2] = 0

        # Convert modified euler angles back to quaternions
        current_quat_modified = R.from_euler('zyx', current_euler, degrees=True).as_quat()
        init_quat_modified = R.from_euler('zyx', init_euler, degrees=True).as_quat()

        # Convert back to tensor
        current_quat_modified = torch.tensor(current_quat_modified, device=env.device)
        init_quat_modified = torch.tensor(init_quat_modified, device=env.device)

        # Compute the dot product of the two quaternions
        dot_product = torch.sum(current_quat_modified * init_quat_modified, dim=1)

        # The absolute value of the dot product between two unit quaternions gives the cosine of half the angle
        # between the two rotations. We'll use this to get the angle difference.
        angle_diff = 2 * torch.acos(torch.clamp(dot_product, -1.0, 1.0))

        # Normalize the angle to [0, 1]
        normalized_angle_diff = angle_diff / torch.tensor([math.pi], device=env.device)

        return -normalized_angle_diff

    def picking_object_success(self, env: PickEnv, threshold: float):
        return torch.where(env.object.data.root_pos_w[:, 2] > threshold, 1.0, 0.0)

    def green_pixel_reward(self, env: PickEnv, thresholds: list):
        previous_camera_tensor = env.previous_camera_tensor
        # 将扁平化的tensor重塑为[batch, width*height, 3]的形状
        reshaped_tensor = previous_camera_tensor.view(previous_camera_tensor.shape[0], -1, 3)
        # 定义绿色的上下界范围
        lower_bound = torch.tensor([0, 0.698, 0.149], device=env.device)
        upper_bound = torch.tensor([0, 0.251, 0.01], device=env.device)
        # 计算中心值
        center_green = (lower_bound + upper_bound) / 2.0
        # 计算与中心值的距离
        distance_from_center = torch.abs(reshaped_tensor - center_green)
        # 确保阈值和数据在相同的设备上
        thresholds_tensor = torch.tensor(thresholds, device=env.device)
        # 检查是否所有通道的值都在相应的阈值以下
        is_approx_green = (distance_from_center <= thresholds_tensor).all(dim=-1)
        # 计算每个环境的绿色像素数量
        green_pixel_count = is_approx_green.sum(dim=-1)
        # 获取图像的总像素数
        total_pixels = env.cfg.camera.width * env.cfg.camera.height
        # 归一化green_pixel_count
        normalized_green_pixel_count = green_pixel_count.float() / total_pixels

        return normalized_green_pixel_count

    def reaching_object_success(self, env: PickEnv, threshold: float):
        # 提取左右两个finger的位置
        left_finger_pos = env.robot.data.tool_sites_state_w[:, 0, :3] - env.envs_positions
        right_finger_pos = env.robot.data.tool_sites_state_w[:, 1, :3] - env.envs_positions
        # 计算EE center的位置（两个finger位置的中点）
        ee_center_pos = (left_finger_pos + right_finger_pos) / 2
        # 提取object位置
        object_pos = env.object.data.root_pos_w - env.envs_positions
        # 计算EE center到object的距离
        distance = torch.norm(ee_center_pos - object_pos, dim=1)
        # print("distance", distance)
        # print("is_success", torch.where(distance < threshold, 1.0, 0.0))
        return torch.where(distance < threshold, 1.0, 0.0)
    
