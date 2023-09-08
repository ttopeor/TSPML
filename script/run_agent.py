import argparse
import asyncio
import websockets
import queue
import json
from functools import partial

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

action_queue = queue.Queue()
env = None
current_talker = None
current_listener = []


async def ws_server():
    server = await websockets.serve(handle_connection, 'localhost', 8765)
    await server.wait_closed()


async def handle_connection(websocket, path):
    global current_talker, current_listener, env

    # First, we expect the client to send its type: 'talker' or 'listener'
    client_type = await websocket.recv()

    if client_type == 'talker':
        if current_talker:  # check if a talker is already connected
            await websocket.send("Another talker is already connected.")
            return
        current_talker = websocket
    elif client_type == 'listener':
        current_listener.append(websocket)
    else:
        print(f"Unknown client type: {client_type}")
        return

    print(f"{client_type} connected.")
    print(f"Current listener number: {len(current_listener)}")

    try:
        if client_type == 'talker':
            while True:
                actions_data = await websocket.recv()
                actions_list = json.loads(actions_data)
                if (len(actions_list) == env.num_envs):
                    actions = torch.tensor(actions_list, dtype=torch.float32, device=env.device)
                    action_queue.put(actions)  # Put actions into queue for processing
                else:
                    print(f"Require {env.num_envs} Robot actions, but {len(actions_list)} actions received.")
        if client_type == 'listener':
            while True:
                await asyncio.sleep(1)

    except websockets.ConnectionClosed:
        if client_type == 'talker':
            current_talker = None
        elif client_type == 'listener' and websocket in current_listener:
            current_listener.remove(websocket)
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
    global env
    env_cfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg, headless=args_cli.headless)
    env.reset()

    # Initialize last_actions as zeros
    last_actions = torch.zeros((env.num_envs, env.action_space.shape[0]), device=env.device)

    # Start websocket server in a separate thread
    import threading
    websocket_thread = threading.Thread(target=websocket_thread_function)
    websocket_thread.start()

    # Create and set a new event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Main loop to simulate environment
    while simulation_app.is_running():
        if not action_queue.empty():
            actions = action_queue.get()  # Get actions from queue
            last_actions = actions
        else:
            actions = last_actions
        _, _, _, _ = env.step(actions)

        if current_listener:
            data_dict = {
                "arm_dof_pos": env.robot.data.arm_dof_pos.cpu().numpy().tolist(),
                "arm_dof_vel": env.robot.data.arm_dof_vel.cpu().numpy().tolist(),
                "arm_dof_acc": env.robot.data.arm_dof_acc.cpu().numpy().tolist()
            }
            json_data = json.dumps(data_dict)  # Convert list to JSON string

            listeners_to_remove = []

            for listener in current_listener:
                successful = loop.run_until_complete(safe_send(listener, json_data))
                if not successful:
                    listeners_to_remove.append(listener)

            # Remove disconnected listeners
            for listener in listeners_to_remove:
                current_listener.remove(listener)
                print(f"listener removed: {len(listeners_to_remove)}")
                print(f"Current listener number: {len(current_listener)}")

        # check if simulator is stopped
        if env.unwrapped.sim.is_stopped():
            break

    # close the simulator
    env.close()
    simulation_app.close()
    
    # Close the event loop
    loop.close()


if __name__ == "__main__":
    main()
