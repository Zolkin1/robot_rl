"""Visualize an arbitrary IsaacLab terrain config in Isaac Sim.

Pass any ``TerrainGeneratorCfg`` (or a single ``SubTerrainBaseCfg``) as a
``module:attribute`` import path. The script spawns the terrain through
``TerrainImporter`` and drops spheres on the env origins so you can verify
both the geometry and the spawn heights.

Examples:

    # robot_rl custom stair generator
    ./isaaclab.sh -p scripts/visualize_terrain.py \
        --terrain=robot_rl.tasks.manager_based.robot_rl.terrains.config.terrain_cfgs:CUSTOM_STAIR_CFG

    # one progressive stairs sub-terrain (auto-wrapped into a generator)
    ./isaaclab.sh -p scripts/visualize_terrain.py \
        --terrain=robot_rl.tasks.manager_based.robot_rl.terrains.trimesh.stair_cfg:MeshProgressiveXStairsTerrainCfg \
        --num_rows=4 --num_cols=4 --color_scheme=random

    # IsaacLab's stock rough cfg as a sanity check
    ./isaaclab.sh -p scripts/visualize_terrain.py \
        --terrain=isaaclab.terrains.config.rough:ROUGH_TERRAINS_CFG
"""

from __future__ import annotations

import argparse
import importlib
from typing import Any

from isaaclab.app import AppLauncher

# -- argparse ---------------------------------------------------------------
parser = argparse.ArgumentParser(description="Visualize an arbitrary terrain config.")
parser.add_argument(
    "--terrain",
    type=str,
    required=True,
    help="Import path to a TerrainGeneratorCfg or SubTerrainBaseCfg, formatted as 'module:attribute'.",
)
parser.add_argument(
    "--color_scheme", type=str, default="none", choices=["random", "none"],
    help="Color scheme passed to the terrain generator. 'none' = plain mesh (default).",
)
parser.add_argument("--curriculum", action="store_true", default=False,
                    help="Sort sub-terrain difficulty along rows.")
parser.add_argument("--num_rows", type=int, default=None, help="Override generator num_rows.")
parser.add_argument("--num_cols", type=int, default=None, help="Override generator num_cols.")
parser.add_argument("--num_envs", type=int, default=512, help="Number of probe spheres to spawn.")
parser.add_argument("--no_balls", action="store_true", default=False,
                    help="Skip dropping spheres (terrain only).")
parser.add_argument("--sub_terrain_size", type=float, nargs=2, default=(8.0, 8.0),
                    metavar=("X", "Y"),
                    help="Tile size used when auto-wrapping a single SubTerrainBaseCfg.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Kit before importing anything that needs the simulation app.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# -- imports that require the running app -----------------------------------
import numpy as np
from isaacsim.core.cloner import GridCloner

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.terrains.terrain_importer import TerrainImporter
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from robot_rl.tasks.manager_based.robot_rl.terrains.meta_terrain_generator_cfg import (
    MetaTerrainGeneratorCfg,
)
from robot_rl.tasks.manager_based.robot_rl.terrains.meta_terrain_importer_cfg import (
    MetaTerrainImporterCfg,
)


def load_terrain(import_path: str) -> Any:
    """Resolve ``module:attribute`` to either an instance or a class.

    Accepts a ``TerrainGeneratorCfg`` instance, a ``SubTerrainBaseCfg`` instance,
    or a ``SubTerrainBaseCfg`` subclass (which gets instantiated with defaults).
    """
    if ":" not in import_path:
        raise ValueError(f"--terrain must be 'module:attribute', got '{import_path}'")
    module_name, attr_name = import_path.split(":", 1)
    module = importlib.import_module(module_name)
    obj = getattr(module, attr_name)
    if isinstance(obj, type) and issubclass(obj, SubTerrainBaseCfg):
        # passed a class — instantiate with sensible defaults
        return obj(proportion=1.0, size=tuple(args_cli.sub_terrain_size))
    return obj


def to_generator_cfg(obj: Any) -> TerrainGeneratorCfg:
    """Wrap a single sub-terrain in a generator if needed; otherwise return as-is."""
    if isinstance(obj, TerrainGeneratorCfg):
        cfg = obj
    elif isinstance(obj, SubTerrainBaseCfg):
        size = tuple(args_cli.sub_terrain_size)
        cfg = TerrainGeneratorCfg(
            size=size,
            num_rows=args_cli.num_rows or 4,
            num_cols=args_cli.num_cols or 4,
            border_width=0.0,
            sub_terrains={"only": obj.replace(proportion=1.0, size=size)},
        )
    else:
        raise TypeError(
            f"--terrain must resolve to a TerrainGeneratorCfg or SubTerrainBaseCfg, got {type(obj).__name__}"
        )

    overrides: dict[str, Any] = {"color_scheme": args_cli.color_scheme, "curriculum": args_cli.curriculum}
    if args_cli.num_rows is not None:
        overrides["num_rows"] = args_cli.num_rows
    if args_cli.num_cols is not None:
        overrides["num_cols"] = args_cli.num_cols
    return cfg.replace(**overrides)


def fit_camera(sim: SimulationContext, generator_cfg: TerrainGeneratorCfg) -> None:
    """Place the camera so the whole grid is visible."""
    extent_x = generator_cfg.size[0] * generator_cfg.num_rows
    extent_y = generator_cfg.size[1] * generator_cfg.num_cols
    diag = float(np.hypot(extent_x, extent_y))
    eye = (0.6 * diag, 0.6 * diag, 0.5 * diag)
    sim.set_camera_view(eye=eye, target=(0.0, 0.0, 0.0))


def spawn_probe_balls(sim: SimulationContext, importer: TerrainImporter) -> None:
    """Drop one sphere above each env spawn origin.

    Reflects exactly where the training-time envs would spawn. Border columns
    are absent because the meta importer already excludes them from
    ``env_origins`` upstream.
    """
    physics_material_cfg = sim_utils.RigidBodyMaterialCfg(
        static_friction=0.2, dynamic_friction=1.0, restitution=0.0,
    )
    visual_material_cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.4, 1.0))
    sphere_cfg = sim_utils.MeshSphereCfg(
        radius=0.1,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        visual_material=visual_material_cfg,
        physics_material=physics_material_cfg,
    )

    targets = importer.env_origins.clone()
    num_balls = int(targets.shape[0])

    cloner = GridCloner(spacing=2.0, stage=sim.stage)
    cloner.define_base_env("/World/envs")
    sim.stage.DefinePrim("/World/envs/env_0", "Xform")
    sphere_cfg.func("/World/envs/env_0/ball", sphere_cfg, translation=(0.0, 0.0, 0.5))

    paths = cloner.generate_paths("/World/envs/env", num_paths=num_balls)
    cloner.clone(source_prim_path="/World/envs/env_0", prim_paths=paths, replicate_physics=True)
    cloner.filter_collisions(
        sim.cfg.physics_prim_path, "/World/collisions",
        prim_paths=paths, global_paths=["/World/ground"],
    )

    targets[:, 2] += 0.5
    xform_view = sim_utils.XformPrimView("/World/envs/env_.*/ball")
    xform_view.set_world_poses(positions=targets)


def main() -> None:
    """Entry point: build the terrain, spawn balls, run the sim loop."""
    raw = load_terrain(args_cli.terrain)
    generator_cfg = to_generator_cfg(raw)

    sim = SimulationContext(SimulationCfg())
    fit_camera(sim, generator_cfg)

    importer_cls = (
        MetaTerrainImporterCfg if isinstance(generator_cfg, MetaTerrainGeneratorCfg)
        else terrain_gen.TerrainImporterCfg
    )
    importer_cfg = importer_cls(
        num_envs=max(args_cli.num_envs, 1), env_spacing=3.0,
        prim_path="/World/ground", max_init_terrain_level=None,
        terrain_type="generator", terrain_generator=generator_cfg,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
    )
    # Use ``class_type`` so the meta importer is resolved through the same
    # string-based path the env uses (no stale runtime import).
    importer = importer_cfg.class_type(importer_cfg)

    # Sky dome — same HDRI the play envs use (see ``G1ClfTrackingSceneCfg.sky_light``).
    sky_cfg = sim_utils.DomeLightCfg(
        intensity=750.0,
        texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
    )
    sky_cfg.func("/World/skyLight", sky_cfg)

    if not args_cli.no_balls:
        spawn_probe_balls(sim, importer)

    sim.reset()
    while simulation_app.is_running() and not sim.is_stopped():
        sim.step()


if __name__ == "__main__":
    main()
    simulation_app.close()
