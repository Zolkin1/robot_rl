
import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'g1_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join("share", package_name, "config"), glob(os.path.join("config", "*.yml"))),
        (os.path.join("share", package_name, "config"), glob(os.path.join("config", "*.obk"))),
        (os.path.join("share", package_name, "resource/policies"), glob(os.path.join("resource/policies", "*.pt"))),  # Get the models..
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zolkin',
    maintainer_email='zach.olkin@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "g1_controller = g1_control.controller:main",
            "g1_estimator = g1_control.estimator:main"
        ],
    },
)