# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to collect demonstrations with Isaac Orbit environments."""

"""Launch Isaac Sim Simulator first."""
import argparse
from omni.isaac.kit import SimulationApp

# add argparse arguments
parser = argparse.ArgumentParser("Welcome to Orbit: Omniverse Robotics Environments!")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Pick-Franka-v0", help="Name of the task.")
parser.add_argument("--num_demos", type=int, default=1024, help="Number of episodes to store in the dataset.")
parser.add_argument("--filename", type=str, default="hdf_dataset", help="Basename of output file.")
args_cli = parser.parse_args()

# launch the simulator
config = {"headless": args_cli.headless}
simulation_app = SimulationApp(config)

"""Rest everything follows."""


import contextlib
import gym
import os
import torch
import asyncio
import websockets
import json
import threading
import numpy as np


from omni.isaac.orbit.devices import Se3Keyboard, Se3SpaceMouse
from omni.isaac.orbit.utils.io import dump_pickle, dump_yaml

import omni.isaac.contrib_envs  # noqa: F401
import omni.isaac.orbit_envs  # noqa: F401
from omni.isaac.orbit_envs.utils.data_collector import RobomimicDataCollector
from omni.isaac.orbit_envs.utils.parse_cfg import parse_env_cfg

env = None
current_talkers = []
current_listeners = []
global_actions = None
listener_env_nums = {}


async def ws_server():
    server = await websockets.serve(handle_connection, 'localhost', 8765)
    await server.wait_closed()


async def handle_connection(websocket, path):
    global current_talkers, current_listeners, env, listener_env_nums

    client_type = await websocket.recv()

    if client_type == 'talker':
        current_talkers.append(websocket)
    elif 'listener' in client_type:   # Assuming listeners will send "listener_X" where X is the env_num
        current_listeners.append(websocket)
        env_num = int(client_type.split('_')[-1])   # Extract the env_num
        listener_env_nums[websocket] = env_num
    else:
        print(f"Unknown client type: {client_type}")
        return

    print(f"{client_type} connected.")
    print(f"Current talkers number: {len(current_talkers)}")
    print(f"Current listener number: {len(current_listeners)}")

    try:
        if client_type == 'talker':
            while True:
                data = await websocket.recv()
                talker_data = json.loads(data)

                if 'env_num' in talker_data and 'desire_pos' in talker_data:
                    env_num = int(talker_data['env_num'])
                    desire_pos = torch.tensor(talker_data['desire_pos'], dtype=torch.float32, device=env.device)

                    if desire_pos.shape[0] == env.action_space.shape[0]:
                        global_actions[env_num] = desire_pos
                    else:
                        print(f"Received data does not match action space shape for env_num {env_num}!")
                        print(f"Desired shape is {env.action_space.shape[0]} but received {desire_pos.shape[0]}")
          
                else:
                    print("Received invalid data from talker!")

        if client_type.startswith('listener'):
            while True:
                await asyncio.sleep(1)

    except websockets.ConnectionClosed as e:
        # Handle the disconnect as before
        if client_type == 'talker' and websocket in current_talkers:
            current_talkers.remove(websocket)
        elif client_type.startswith('listener') and websocket in current_listeners:
            current_listeners.remove(websocket)
        print(f"{client_type} disconnected due to: {str(e)}")

    except Exception as e:
        # Handle other general exceptions
        print(f"An unexpected error occurred: {str(e)}")


def websocket_thread_function():
    new_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(new_loop)
    new_loop.run_until_complete(ws_server())
    new_loop.close()


async def safe_send(listener, json_data):
    try:
        await listener.send(json_data)
    except (websockets.exceptions.ConnectionClosedOK, websockets.exceptions.ConnectionClosedError) as e:
        print(f"WebSocket error: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False
    return True


def listener_server_spin(loop, data):
    global env

    if current_listeners:
        listeners_to_remove = []

        for listener in current_listeners:
            env_num = listener_env_nums.get(listener)
            if env_num is not None:
                data_dict = data[env_num]
                json_data = json.dumps(data_dict)
                successful = loop.run_until_complete(safe_send(listener, json_data))
                if not successful:
                    listeners_to_remove.append(listener)
            else:
                listeners_to_remove.append(listener)  # Remove if no env_num is associated

        for listener in listeners_to_remove:
            current_listeners.remove(listener)
            listener_env_nums.pop(listener, None)  # Remove the listener from the env_num mapping as well
            print(f"listener removed: {len(listeners_to_remove)}")
            print(f"Current listener number: {len(current_listeners)}")


def ros_reset(loop):
    global env
    # get joint states from env
    arm_dof_pos = env.robot.data.arm_dof_pos.cpu().numpy().tolist()
    arm_dof_vel = env.robot.data.arm_dof_vel.cpu().numpy().tolist()
    tool_dof_pos = env.robot.data.tool_dof_pos.cpu().numpy().tolist()
    tool_dof_vel = env.robot.data.tool_dof_vel.cpu().numpy().tolist()
    object_positions = get_object_positions().cpu().numpy().tolist()
    object_orientations = get_object_orientations().cpu().numpy().tolist()
    reset_buf = env.reset_buf.cpu().numpy().tolist()
    reset_buf = [1 for _ in reset_buf]

    data = pre_process_obs(arm_dof_pos, arm_dof_vel, tool_dof_pos, tool_dof_vel, object_positions, object_orientations, reset_buf)

    if current_listeners:
        listeners_to_remove = []

        for listener in current_listeners:
            env_num = listener_env_nums.get(listener)
            if env_num is not None:
                data_dict = data[env_num]
                json_data = json.dumps(data_dict)
                successful = loop.run_until_complete(safe_send(listener, json_data))
                if not successful:
                    listeners_to_remove.append(listener)
            else:
                listeners_to_remove.append(listener)  # Remove if no env_num is associated

        for listener in listeners_to_remove:
            current_listeners.remove(listener)
            listener_env_nums.pop(listener, None)  # Remove the listener from the env_num mapping as well
            print(f"listener removed: {len(listeners_to_remove)}")
            print(f"Current listener number: {len(current_listeners)}")


def pre_process_actions(offsets, action_raw: torch.Tensor) -> torch.Tensor:
    """Pre-process actions for the environment."""
    # to convert the action from franka space to RL space
    offsets_tensor = torch.from_numpy(offsets).to(action_raw.dtype).to(env.device)
    # Subtract the offsets only from the first 7 elements
    offset_action_first7 = action_raw[:, :7] - offsets_tensor
    # Concatenate the result with the last element of action_raw
    offset_action = torch.cat([offset_action_first7, action_raw[:, 7:]], dim=1)
    
    return offset_action


def pre_process_obs(arm_dof_pos, arm_dof_vel, tool_dof_pos, tool_dof_vel, object_positions, object_orientations, reset_buf):
    """Pre-process joint_states for the environment."""
    data = []
    for num in range(env.num_envs):
        item_data = {
                    "arm_dof_pos": arm_dof_pos[num],
                    "arm_dof_vel": arm_dof_vel[num],
                    "tool_dof_pos": tool_dof_pos[num],
                    "tool_dof_vel": tool_dof_vel[num],
                    "object_positions": object_positions[num],
                    "object_orientations": object_orientations[num],
                    "reset_buf": int(reset_buf[num]),
                }
        data.append(item_data)
    return data


def get_joints_offsets():

    offsets_panda_shoulder = env.cfg.robot.actuator_groups["panda_shoulder"].control_cfg.dof_pos_offset
    offsets_panda_forearm = env.cfg.robot.actuator_groups["panda_forearm"].control_cfg.dof_pos_offset

    joint1_offset = offsets_panda_shoulder["panda_joint1"]
    joint2_offset = offsets_panda_shoulder["panda_joint2"]
    joint3_offset = offsets_panda_shoulder["panda_joint3"]
    joint4_offset = offsets_panda_shoulder["panda_joint4"]
    joint5_offset = offsets_panda_forearm["panda_joint5"]
    joint6_offset = offsets_panda_forearm["panda_joint6"]
    joint7_offset = offsets_panda_forearm["panda_joint7"]

    offsets = np.array([joint1_offset, joint2_offset, joint3_offset, joint4_offset, joint5_offset, joint6_offset, joint7_offset])
    return offsets


def initial_global_actions():
    global global_actions
    dof_pos, _ = env.robot.get_default_dof_state()
    dof_pos[0, -2] = 1.0 if dof_pos[0, -2] > 0.035 else -1.0
    global_actions = dof_pos[:, :-1]
    return


def get_object_orientations():
    """Current object orientation."""
    # make the first element positive
    quat_w = env.object.data.root_quat_w
    quat_w[quat_w[:, 0] < 0] *= -1
    return quat_w


def get_object_positions():
    """Current object position."""
    return env.object.data.root_pos_w - env.envs_positions


def main():
    global env, global_actions
    """Collect demonstrations from the environment using teleop interfaces."""
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs)
    # modify configuration
    env_cfg.terminations.episode_timeout = False
    env_cfg.terminations.is_success = True
    env_cfg.observations.return_dict_obs_in_group = True

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg, headless=args_cli.headless)

    # specify directory for logging experiments
    log_dir = os.path.join("./logs/robomimic", args_cli.task)
    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)

    # create data-collector
    collector_interface = RobomimicDataCollector(
        env_name=args_cli.task,
        directory_path=log_dir,
        filename=args_cli.filename,
        num_demos=args_cli.num_demos,
        flush_freq=env.num_envs,
    )

    # reset environment
    obs_dict = env.reset()
    # robomimic only cares about policy observations
    obs = obs_dict["policy"]
    # reset interfaces
    
    collector_interface.reset()

    # Websocket
    websocket_thread = threading.Thread(target=websocket_thread_function)
    websocket_thread.start()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    offsets = get_joints_offsets()
    initial_global_actions()

    # simulate environment
    with contextlib.suppress(KeyboardInterrupt):
        while not collector_interface.is_stopped():
            # get joint states from env
            arm_dof_pos = env.robot.data.arm_dof_pos.cpu().numpy().tolist()
            arm_dof_vel = env.robot.data.arm_dof_vel.cpu().numpy().tolist()
            tool_dof_pos = env.robot.data.tool_dof_pos.cpu().numpy().tolist()
            tool_dof_vel = env.robot.data.tool_dof_vel.cpu().numpy().tolist()
            object_positions = get_object_positions().cpu().numpy().tolist()
            object_orientations = get_object_orientations().cpu().numpy().tolist()
            reset_buf = env.success_counters.cpu().numpy().tolist()
            teacher_obs = pre_process_obs(arm_dof_pos, arm_dof_vel, tool_dof_pos, tool_dof_vel, object_positions, object_orientations, reset_buf)

            # send out joint states to websocket
            listener_server_spin(loop, teacher_obs)

            # get action_raw from websocket
            # compute actions based on environment offsets
            actions = pre_process_actions(offsets, global_actions)
            
            # -- obs
            for key, value in obs.items():
                collector_interface.add(f"obs/{key}", value)
            # -- actions
            collector_interface.add("actions", actions)
            # perform action on environment
            obs_dict, rewards, dones, info = env.step(actions)
            # check that simulation is stopped or not
            if env.unwrapped.sim.is_stopped():
                break
            # robomimic only cares about policy observations
            obs = obs_dict["policy"]
            # store signals from the environment
            # -- next_obs
            for key, value in obs.items():
                collector_interface.add(f"next_obs/{key}", value.cpu().numpy())
            # -- rewards
            collector_interface.add("rewards", rewards)
            # -- dones
            collector_interface.add("dones", dones)
            # -- is-success label
            try:
                collector_interface.add("success", info["is_success"])
            except KeyError:
                raise RuntimeError(
                    f"Only goal-conditioned environment supported. No attribute named 'is_success' found in {list(info.keys())}."
                )
            # flush data from collector for successful environments
            reset_env_ids_flush = dones.nonzero(as_tuple=False).squeeze(-1)
            collector_interface.flush(reset_env_ids_flush)

    # close the simulator
    collector_interface.close()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
