import argparse
import asyncio
import websockets
import json
from functools import partial
import threading

import gym
import torch
from omni.isaac.kit import SimulationApp

# add argparse arguments
parser = argparse.ArgumentParser("Welcome to Orbit: Omniverse Robotics Environments!")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Pick-Franka-v0", help="Name of the task.")
args_cli = parser.parse_args()

# launch the simulator
config = {"headless": args_cli.headless}
simulation_app = SimulationApp(config)


import omni.isaac.orbit_envs  # noqa: F401
from omni.isaac.orbit_envs.utils import parse_env_cfg

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
                    env_num = talker_data['env_num']
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

    except websockets.ConnectionClosed:
        if client_type == 'talker' and websocket in current_talkers:
            current_talkers.remove(websocket)
        elif client_type == 'listener' and websocket in current_listeners:
            current_listeners.remove(websocket)
        print(f"{client_type} disconnected.")


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


def main():
    global env, global_actions

    # Your environment setup code here
    env_cfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg, headless=args_cli.headless)
    env.reset()

    # Initialize the global action tensor now
    dof_pos, _ = env.robot.get_default_dof_state()
    dof_pos[0, -2] = 1.0 if dof_pos[0, -2] > 0.035 else -1.0
    dof_pos = dof_pos[:, :-1]
    global_actions = dof_pos

    websocket_thread = threading.Thread(target=websocket_thread_function)
    websocket_thread.start()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    while simulation_app.is_running():
        _, _, _, _ = env.step(global_actions)
        if current_listeners:
            listeners_to_remove = []

            for listener in current_listeners:
                env_num = listener_env_nums.get(listener)
                if env_num is not None:
                    data_dict = {
                        "arm_dof_pos": env.robot.data.arm_dof_pos[env_num].cpu().numpy().tolist(),
                        "arm_dof_vel": env.robot.data.arm_dof_vel[env_num].cpu().numpy().tolist(),
                        "tool_dof_pos": env.robot.data.tool_dof_pos[env_num].cpu().numpy().tolist(),
                        "tool_dof_vel": env.robot.data.tool_dof_vel[env_num].cpu().numpy().tolist(),
                        "reset_buf": int(env.reset_buf[env_num].item()),
                    }
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

        if env.unwrapped.sim.is_stopped():
            break

    env.close()
    simulation_app.close()
    loop.close()


if __name__ == "__main__":
    main()
