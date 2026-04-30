# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from typing_extensions import override

import isaaclab.sim as sim_utils
from isaaclab.terrains.terrain_importer import TerrainImporter

if TYPE_CHECKING:
    from .meta_terrain_importer_cfg import MetaTerrainImporterCfg


class MetaTerrainImporter(TerrainImporter):
    r"""A class to handle terrain meshes and import them into the simulator.

    We assume that a terrain mesh comprises of sub-terrains that are arranged in a grid with
    rows ``num_rows`` and columns ``num_cols``. The terrain origins are the positions of the sub-terrains
    where the robot should be spawned.

    Based on the configuration, the terrain importer handles computing the environment origins from the sub-terrain
    origins. In a typical setup, the number of sub-terrains (:math:`num\_rows \times num\_cols`) is smaller than
    the number of environments (:math:`num\_envs`). In this case, the environment origins are computed by
    sampling the sub-terrain origins.

    If a curriculum is used, it is possible to update the environment origins to terrain origins that correspond
    to a harder difficulty. This is done by calling :func:`update_terrain_levels`. The idea comes from game-based
    curriculum. For example, in a game, the player starts with easy levels and progresses to harder levels.

    This importer supports inter-column borders - flat sections between terrain segments where robots
    should not spawn.
    """

    terrain_prim_paths: list[str]
    """A list containing the USD prim paths to the imported terrains."""

    terrain_origins: torch.Tensor | None
    """The origins of the sub-terrains in the added terrain mesh. Shape is (num_rows, num_cols, 3).

    If terrain origins is not None, the environment origins are computed based on the terrain origins.
    Otherwise, the environment origins are computed based on the grid spacing.
    """

    terrain_meta_data: dict
    """A dictionary containing meta data about the terrain."""

    env_origins: torch.Tensor
    """The origins of the environments. Shape is (num_envs, 3)."""

    def __init__(self, cfg: MetaTerrainImporterCfg):
        """Initialize the terrain importer.

        Args:
            cfg: The configuration for the terrain importer.

        Raises:
            ValueError: If input terrain type is not supported.
            ValueError: If terrain type is 'generator' and no configuration provided for ``terrain_generator``.
            ValueError: If terrain type is 'usd' and no configuration provided for ``usd_path``.
            ValueError: If terrain type is 'usd' or 'plane' and no configuration provided for ``env_spacing``.
        """
        # check that the config is valid
        cfg.validate()
        # store inputs
        self.cfg = cfg
        self.device = sim_utils.SimulationContext.instance().device  # type: ignore

        # create buffers for the terrains
        self.terrain_prim_paths = list()
        self.terrain_origins = None
        self.env_origins = None  # assigned later when `configure_env_origins` is called
        # private variables
        self._terrain_flat_patches = dict()
        self._spawnable_columns = None
        self._actual_terrain_cols = None

        # auto-import the terrain based on the config
        if self.cfg.terrain_type == "generator":
            # check config is provided
            if self.cfg.terrain_generator is None:
                raise ValueError("Input terrain type is 'generator' but no value provided for 'terrain_generator'.")
            # generate the terrain
            terrain_generator = self.cfg.terrain_generator.class_type(
                cfg=self.cfg.terrain_generator, device=self.device
            )
            self.import_mesh("terrain", terrain_generator.terrain_mesh)
            # configure the terrain origins based on the terrain generator
            self.configure_env_origins(terrain_generator.terrain_origins, terrain_generator.terrain_metadata)
            # refer to the flat patches
            self._terrain_flat_patches = terrain_generator.flat_patches
            self.terrain_meta_data = terrain_generator.terrain_metadata
        else:
            raise ValueError(f"MetaTerrain type '{self.cfg.terrain_type}' not available.")

        # set initial state of debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

        # post-process metadata
        self._post_process_meta_data()

    def _compute_spawnable_columns(self, terrain_metadata: dict) -> torch.Tensor:
        """Get indices of columns where robots can spawn (non-border columns).

        Args:
            terrain_metadata: Dictionary mapping (row, col) to metadata.

        Returns:
            Tensor of spawnable column indices.
        """
        num_rows, num_cols = self.terrain_origins.shape[:2]
        spawnable_cols = []

        for c in range(num_cols):
            is_border_col = False
            # Check if any cell in this column is marked as a border
            for r in range(num_rows):
                if (r, c) in terrain_metadata:
                    if terrain_metadata[(r, c)].get("is_border", False):
                        is_border_col = True
                        break
            if not is_border_col:
                spawnable_cols.append(c)

        return torch.tensor(spawnable_cols, device=self.device, dtype=torch.long)

    def configure_env_origins(self, origins: torch.Tensor | None = None, terrain_metadata: dict | None = None):
        """Configure the origins of the environments, excluding border columns.

        Args:
            origins: The origins of the sub-terrains. Shape is (num_rows, num_cols, 3).
            terrain_metadata: Dictionary mapping (row, col) to metadata, used to identify borders.
        """
        if origins is not None:
            # convert to tensor if needed
            if isinstance(origins, torch.Tensor):
                self.terrain_origins = origins.to(self.device, dtype=torch.float)
            else:
                self.terrain_origins = torch.from_numpy(origins).to(self.device, dtype=torch.float)

            # Get spawnable columns (non-border)
            if terrain_metadata is not None:
                self._spawnable_columns = self._compute_spawnable_columns(terrain_metadata)
            else:
                # No metadata - all columns are spawnable
                self._spawnable_columns = torch.arange(
                    self.terrain_origins.shape[1], device=self.device, dtype=torch.long
                )

            num_spawnable_cols = len(self._spawnable_columns)
            if num_spawnable_cols == 0:
                raise ValueError("No spawnable columns available (all are borders)")

            # Compute environment origins using only spawnable columns
            self.env_origins = self._compute_env_origins_curriculum_with_borders(
                self.cfg.num_envs, self.terrain_origins, self._spawnable_columns
            )
        else:
            self.terrain_origins = None
            if self.cfg.env_spacing is None:
                raise ValueError("Environment spacing must be specified for grid-like origins.")
            self.env_origins = self._compute_env_origins_grid(self.cfg.num_envs, self.cfg.env_spacing)

    def _compute_env_origins_curriculum_with_borders(
        self, num_envs: int, origins: torch.Tensor, spawnable_columns: torch.Tensor
    ) -> torch.Tensor:
        """Compute environment origins using only spawnable columns.

        Args:
            num_envs: Number of environments.
            origins: Tensor of terrain origins with shape (num_rows, num_cols, 3).
            spawnable_columns: Tensor of column indices where spawning is allowed.

        Returns:
            Tensor of environment origins with shape (num_envs, 3).
        """
        num_rows = origins.shape[0]
        num_spawnable_cols = len(spawnable_columns)

        # Maximum initial level
        if self.cfg.max_init_terrain_level is None:
            max_init_level = num_rows - 1
        else:
            max_init_level = min(self.cfg.max_init_terrain_level, num_rows - 1)

        self.max_terrain_level = num_rows

        # terrain_levels: row indices (0 to num_rows-1)
        self.terrain_levels = torch.randint(0, max_init_level + 1, (num_envs,), device=self.device)

        # terrain_types: index into spawnable_columns array (0 to num_spawnable_cols-1)
        self.terrain_types = torch.div(
            torch.arange(num_envs, device=self.device), (num_envs / num_spawnable_cols), rounding_mode="floor"
        ).to(torch.long)

        # Map terrain_types to actual column indices
        self._actual_terrain_cols = spawnable_columns[self.terrain_types]

        env_origins = torch.zeros(num_envs, 3, device=self.device)
        env_origins[:] = origins[self.terrain_levels, self._actual_terrain_cols]
        return env_origins

    @override
    def update_env_origins(self, env_ids: torch.Tensor, move_up: torch.Tensor, move_down: torch.Tensor):
        """Update the environment origins based on terrain levels."""
        if self.terrain_origins is None:
            return

        # Update terrain level for the envs
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # robots that solve the last level are sent to a random one
        # the minimum level is zero
        self.terrain_levels[env_ids] = torch.where(
            self.terrain_levels[env_ids] >= self.max_terrain_level,
            torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
            torch.clip(self.terrain_levels[env_ids], 0),
        )

        # Update the env origins using actual column indices (which exclude borders)
        self.env_origins[env_ids] = self.terrain_origins[
            self.terrain_levels[env_ids],
            self._actual_terrain_cols[env_ids]
        ]

    def _post_process_meta_data(self):
        pass
