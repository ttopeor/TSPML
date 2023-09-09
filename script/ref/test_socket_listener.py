import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import asyncio
import websockets
import websockets.exceptions
import json
import time
import os

ENV_NUM = 0
os.environ['ROS_DOMAIN_ID'] = str(ENV_NUM)


class JointStatePublisher(Node):
    def __init__(self):
        super().__init__('joint_state_publisher')
        self.publisher = self.create_publisher(JointState, '/isaac_joint_states', 10)
        self.start_time = time.time()

    def publish_data(self, data):
        joint_state_msg = JointState()
        joint_state_msg.name = ["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7", "panda_finger_joint1", "panda_finger_joint2"]

        # Filling time stamps
        current_time = time.time() - self.start_time  # Calculate time elapsed since the start
        joint_state_msg.header.stamp.sec = int(current_time)  # Extracts the second part
        joint_state_msg.header.stamp.nanosec = int((current_time - int(current_time)) * 1e9)  # Extracts the nanosecond part

        # Here I'm filling only the first 7 positions, velocities, and efforts with your data
        joint_state_msg.position = data['arm_dof_pos'] + data['tool_dof_pos']  # assuming last two are 0 for fingers
        joint_state_msg.velocity = data['arm_dof_vel'] + data['tool_dof_vel']
        # joint_state_msg.effort = data['arm_dof_acc'] + data['tool_dof_acc']  # Assuming the acc is actually effort data. If not, replace this

        self.publisher.publish(joint_state_msg)


async def listener_client(env_num, publisher_node):
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as websocket:
        # Identify itself as a listener with its env_num
        await websocket.send(f"listener_{env_num}")

        try:
            while True:
                # Continuously receive data from the server
                data_str = await websocket.recv()
                data = json.loads(data_str)  # Convert JSON string to Python dictionary
                
                # Publish the received data using ROS2 publisher
                publisher_node.publish_data(data)

                # print(f"Received data from server for env {env_num}: {data}")

        except websockets.exceptions.ConnectionClosedOK:
            print("Connection closed by the server.")

        except Exception as e:
            print(f"Unexpected error: {e}")

rclpy.init()
publisher_node = JointStatePublisher()

# Run the listener client for the specified ENV_NUM
asyncio.get_event_loop().run_until_complete(listener_client(ENV_NUM, publisher_node))

# Keep ROS2 node running
rclpy.spin(publisher_node)
