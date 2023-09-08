from setuptools import find_packages
from setuptools import setup

setup(
    name='rclpy',
    version='3.3.9',
    packages=find_packages(
        include=('rclpy', 'rclpy.*')),
)
