"""Recursive contact sensor: lightweight config module.

This module is safe to import before ``SimulationApp`` is created. It only contains
the ``RecursiveContactSensorCfg`` class with no ``pxr``/``omni``/``isaaclab_physx`` imports.

The actual sensor class and the monkey-patch for ``activate_contact_sensors`` live in
``_recursive_contact_sensor_impl.py``, which must be imported after ``SimulationApp``
is running. To ensure the patch is applied before the scene is created, add this line
to your train/play scripts after ``launch_simulation`` but before ``gym.make``::

    import robot_rl.sensors._recursive_contact_sensor_impl  # noqa: F401
"""

from __future__ import annotations

from isaaclab.sensors import ContactSensorCfg as _BaseContactSensorCfg
from isaaclab.utils import configclass


@configclass
class RecursiveContactSensorCfg(_BaseContactSensorCfg):
    """Configuration for the recursive contact sensor.

    Uses a string ``class_type`` to defer the sensor class import until after
    ``SimulationApp`` has been created.
    """

    class_type: type | str = (
        "robot_rl.sensors._recursive_contact_sensor_impl:RecursiveContactSensor"
    )
