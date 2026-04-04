"""Recursive contact sensor implementation.

This module requires ``SimulationApp`` to be running (imports ``pxr`` and ``isaaclab_physx``).
It is loaded lazily via the string ``class_type`` in ``RecursiveContactSensorCfg``.

**Important**: This module also monkey-patches ``activate_contact_sensors`` at import time
so that ALL rigid bodies in an articulation get ``PhysxContactReportAPI`` (not just the root).
To ensure the patch is applied before the scene is created, import this module explicitly
in your train/play scripts after ``launch_simulation`` but before ``gym.make``::

    import robot_rl.sensors._recursive_contact_sensor_impl  # noqa: F401
"""

from __future__ import annotations

import logging

from pxr import Usd, UsdPhysics

import isaaclab.sim as sim_utils
import isaaclab.sim.schemas.schemas as _schemas
from isaaclab.sim.utils.prims import safe_set_attribute_on_usd_prim
from isaaclab_physx.physics import PhysxManager as SimulationManager
from isaaclab_physx.sensors.contact_sensor import ContactSensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Patch: recursive activate_contact_sensors
# ---------------------------------------------------------------------------

def _activate_contact_sensors_recursive(
    prim_path: str, threshold: float = 0.0, stage: Usd.Stage | None = None
) -> bool:
    """Apply PhysxContactReportAPI to ALL rigid bodies under *prim_path*, recursively.

    Unlike the default ``activate_contact_sensors`` in IsaacLab, this version continues
    recursing into children of rigid bodies. This is necessary for articulations whose
    USD hierarchy nests link bodies under each other (common with URDF converters).

    Args:
        prim_path: The prim path under which to search and prepare contact sensors.
        threshold: The threshold for the contact sensor. Defaults to 0.0.
        stage: The stage where to find the prim. Defaults to None (current stage).

    Returns:
        True if at least one contact sensor was added.

    Raises:
        ValueError: If the prim path is not valid or no rigid bodies are found.
    """
    if stage is None:
        stage = sim_utils.get_current_stage()

    prim: Usd.Prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim path '{prim_path}' is not valid.")

    num_contact_sensors = 0
    all_prims = [prim]
    while all_prims:
        child_prim = all_prims.pop(0)
        if child_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            child_applied = child_prim.GetAppliedSchemas()
            if "PhysxRigidBodyAPI" not in child_applied:
                child_prim.AddAppliedSchema("PhysxRigidBodyAPI")
            safe_set_attribute_on_usd_prim(
                child_prim, "physxRigidBody:sleepThreshold", 0.0, camel_case=False
            )
            if "PhysxContactReportAPI" not in child_applied:
                child_prim.AddAppliedSchema("PhysxContactReportAPI")
            safe_set_attribute_on_usd_prim(
                child_prim, "physxContactReport:threshold", threshold, camel_case=False
            )
            num_contact_sensors += 1
        # Always recurse — articulation links are rigid bodies with rigid-body children
        all_prims += child_prim.GetChildren()

    if num_contact_sensors == 0:
        raise ValueError(
            f"No contact sensors added to the prim: '{prim_path}'. No rigid bodies found."
        )
    logger.info(
        "[activate_contact_sensors_recursive] Applied PhysxContactReportAPI to %d bodies "
        "under %s",
        num_contact_sensors,
        prim_path,
    )
    return True


# Apply the monkey-patch at import time (this module is imported after SimulationApp).
_schemas.activate_contact_sensors = _activate_contact_sensors_recursive
logger.info("[RecursiveContactSensor] Patched activate_contact_sensors for recursive traversal")


# ---------------------------------------------------------------------------
# Recursive body discovery for the sensor
# ---------------------------------------------------------------------------

def _find_contact_bodies_recursive(root_prim: Usd.Prim) -> list[tuple[str, str]]:
    """Recursively find all prims with PhysxContactReportAPI under *root_prim*.

    Args:
        root_prim: The USD prim to start searching from.

    Returns:
        A list of ``(body_name, full_prim_path)`` tuples for each matching body.
    """
    results: list[tuple[str, str]] = []
    stack = list(root_prim.GetAllChildren())
    while stack:
        prim = stack.pop()
        if "PhysxContactReportAPI" in prim.GetAppliedSchemas():
            prim_path = prim.GetPath().pathString
            body_name = prim_path.rsplit("/", 1)[-1]
            results.append((body_name, prim_path))
        # Always recurse — rigid body children may also be rigid bodies in articulations
        stack.extend(prim.GetAllChildren())
    return results


# ---------------------------------------------------------------------------
# Sensor class
# ---------------------------------------------------------------------------

class RecursiveContactSensor(ContactSensor):
    """Contact sensor that recursively discovers rigid bodies in deeply nested USD trees.

    Drop-in replacement for the PhysX ``ContactSensor``. The only difference is in
    ``_initialize_impl``: instead of matching direct children via ``find_matching_prims``,
    it recursively walks the USD subtree to find all prims with ``PhysxContactReportAPI``.
    """

    def _initialize_impl(self):
        """Initialize the sensor with recursive body discovery."""
        # Call the grandparent (BaseContactSensor -> SensorBase) init,
        # which sets _device, _backend, _parent_prims, _num_envs, etc.
        # We skip ContactSensor._initialize_impl entirely and replace it.
        super(ContactSensor, self)._initialize_impl()

        # Obtain global simulation view
        self._physics_sim_view = SimulationManager.get_physics_sim_view()

        # --- Recursive body discovery (replaces find_matching_prims) ---
        template_prim_path = self._parent_prims[0].GetPath().pathString
        stage = sim_utils.get_current_stage()
        root_prim = stage.GetPrimAtPath(template_prim_path)

        if not root_prim.IsValid():
            raise RuntimeError(
                f"RecursiveContactSensor: parent prim at '{template_prim_path}' is not valid."
            )

        found_bodies = _find_contact_bodies_recursive(root_prim)

        if not found_bodies:
            raise RuntimeError(
                f"RecursiveContactSensor at path '{self.cfg.prim_path}' could not find any "
                "bodies with PhysxContactReportAPI under the parent prim.\n"
                "HINT: Make sure to enable 'activate_contact_sensors' in the corresponding "
                "asset spawn configuration."
            )

        body_names = [name for name, _ in found_bodies]
        body_full_paths = [path for _, path in found_bodies]

        print(
            f"[RecursiveContactSensor] Found {len(body_names)} contact bodies: {body_names}"
        )

        # --- Construct wildcarded prim paths for all envs ---
        # Convert env-0 specific paths to wildcard paths that match all envs.
        env_wildcard_paths = []
        for full_path in body_full_paths:
            wildcard_path = full_path.replace("/env_0/", "/env_*/")
            env_wildcard_paths.append(wildcard_path)

        # --- Create PhysX views ---
        filter_prim_paths_glob = [
            expr.replace(".*", "*") for expr in self.cfg.filter_prim_paths_expr
        ]

        self._body_physx_view = self._physics_sim_view.create_rigid_body_view(
            env_wildcard_paths
        )
        self._contact_view = self._physics_sim_view.create_rigid_contact_view(
            env_wildcard_paths,
            filter_patterns=filter_prim_paths_glob,
            max_contact_data_count=(
                self.cfg.max_contact_data_count_per_prim
                * len(body_names)
                * self._num_envs
            ),
        )

        # Resolve the true count of bodies
        self._num_sensors = self.body_physx_view.count // self._num_envs

        if self._num_sensors != len(body_names):
            raise RuntimeError(
                "RecursiveContactSensor: failed to initialize contact reporter for "
                "specified bodies."
                f"\n\tExpected {len(body_names)} bodies, got {self._num_sensors}."
                f"\n\tBody names: {body_names}"
                f"\n\tWildcard paths: {env_wildcard_paths}"
            )

        # Validate filter paths
        if self.cfg.track_contact_points or self.cfg.track_friction_forces:
            if not self.cfg.filter_prim_paths_expr:
                raise ValueError(
                    "The 'filter_prim_paths_expr' is empty. Please specify a valid "
                    "filter pattern to track "
                    f"{'contact points' if self.cfg.track_contact_points else 'friction forces'}."
                )
            if self.cfg.max_contact_data_count_per_prim < 1:
                raise ValueError(
                    f"The 'max_contact_data_count_per_prim' is "
                    f"{self.cfg.max_contact_data_count_per_prim}. Please set it to a "
                    "value greater than 0 to track "
                    f"{'contact points' if self.cfg.track_contact_points else 'friction forces'}."
                )

        self._create_buffers()
