import asyncio
import websockets
import json
import torch
import threading
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import os

ENV_NUM = 0
os.environ['ROS_DOMAIN_ID'] = str(ENV_NUM)


def start_asyncio_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


# Create a new asyncio loop and run it in a separate thread
asyncio_loop = asyncio.new_event_loop()
threading.Thread(target=start_asyncio_loop, args=(asyncio_loop,)).start()


class TalkerNode(Node):
    def __init__(self, websocket, env_num):
        super().__init__('talker_node')
        self.websocket = websocket
        self.env_num = env_num  # Assign the env_num
        self.subscription = self.create_subscription(
            JointState,
            '/isaac_joint_commands',
            self.listener_callback,
            10
        )
        self.joint_state_combined = JointState()

    def listener_callback(self, msg):
        # Check if the received message is for arm or fingers
        if "panda_joint1" in msg.name:
            self.joint_state_combined.position[:7] = msg.position
        elif "panda_finger_joint1" in msg.name:
            self.joint_state_combined.position[7:] = msg.position

        tensor_data = torch.tensor(self.joint_state_combined.position)

        position_list = tensor_data.numpy().tolist()
        position_list.pop()  # remove the last element from the list
        position_list[-1] = 1.0 if position_list[-1] > 0.035 else -1.0

        # Create a dictionary to send
        data_dict = {
            "env_num": self.env_num,
            "desire_pos": position_list  # Use the modified list
        }
        data_to_send = json.dumps(data_dict)

        asyncio.run_coroutine_threadsafe(self.send_data(data_to_send), asyncio_loop)

    async def send_data(self, data):
        await self.websocket.send(data)
        # print("Sent:", data)


async def talker_client():
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as websocket:
        # Identify itself as a talker
        await websocket.send("talker")

        rclpy.init()
        talker_node = TalkerNode(websocket,env_num=ENV_NUM)

        # Start rclpy.spin in a separate thread
        thread = threading.Thread(target=rclpy.spin, args=(talker_node,))
        thread.start()

        while not websocket.closed:
            await asyncio.sleep(1)

        rclpy.shutdown()
        thread.join()  # Wait for rclpy.spin to finish


# Run the talker client
asyncio.get_event_loop().run_until_complete(talker_client())
