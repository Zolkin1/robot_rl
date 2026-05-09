from __future__ import annotations

import os
import trimesh
import numpy as np
from typing_extensions import override
from isaaclab.utils.io import dump_yaml
from isaaclab.utils.dict import dict_to_md5_hash
from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.terrains.terrain_generator import TerrainGenerator
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.terrains.trimesh.utils import make_plane


class MetaTerrainGenerator(TerrainGenerator):
    """Terrain generator for meta-data terrains."""

    def __init__(self, cfg: TerrainGeneratorCfg, device: str = "cpu"):
        self.terrain_metadata = {}
        # Compute and store border column information before parent init
        self._border_columns, self._content_num_cols = self._compute_border_column_info(cfg)
        super().__init__(cfg, device)

    def _compute_border_column_info(self, cfg: TerrainGeneratorCfg) -> tuple[set[int], int]:
        """Compute which actual column indices are border columns.

        Args:
            cfg: The terrain generator configuration.

        Returns:
            border_columns: Set of column indices that are borders.
            content_num_cols: Number of columns for content (excluding borders).
        """
        if not hasattr(cfg, 'inter_column_borders') or not cfg.inter_column_borders:
            return set(), cfg.num_cols

        # Calculate how many columns each border needs
        border_col_counts = []
        total_border_cols = 0
        for _, width in cfg.inter_column_borders:
            num_cols = int(np.ceil(width / cfg.size[1]))
            border_col_counts.append(num_cols)
            total_border_cols += num_cols

        # Content columns = total - border columns
        content_num_cols = cfg.num_cols - total_border_cols
        if content_num_cols <= 0:
            raise ValueError(
                f"Border columns ({total_border_cols}) exceed or equal total columns ({cfg.num_cols}). "
                "Increase num_cols to accommodate borders."
            )

        # Normalize proportions and get sub-terrain names
        proportions = np.array([sub_cfg.proportion for sub_cfg in cfg.sub_terrains.values()])
        proportions /= np.sum(proportions)
        sub_terrain_names = list(cfg.sub_terrains.keys())

        # Calculate where borders are inserted
        border_columns = set()
        cumulative_offset = 0
        for i, (after_name, _) in enumerate(cfg.inter_column_borders):
            # Find cumulative proportion up to and including after_name
            if after_name not in sub_terrain_names:
                raise ValueError(
                    f"Sub-terrain '{after_name}' not found in sub_terrains. "
                    f"Available: {sub_terrain_names}"
                )
            name_idx = sub_terrain_names.index(after_name)
            cum_prop = np.cumsum(proportions)[name_idx]
            # Content column after which border is inserted (0-indexed)
            content_col_after = int(cum_prop * content_num_cols)
            # Actual column index accounting for previous borders
            actual_start_col = content_col_after + cumulative_offset
            for j in range(border_col_counts[i]):
                border_columns.add(actual_start_col + j)
            cumulative_offset += border_col_counts[i]

        return border_columns, content_num_cols

    def _generate_flat_border_terrain(self) -> tuple[trimesh.Trimesh, np.ndarray, dict]:
        """Generate a flat terrain for border columns.

        Returns:
            mesh: The flat terrain mesh.
            origin: The origin of the terrain.
            meta_data: Metadata marking this as a border.
        """
        # Create a simple flat plane
        mesh = make_plane(self.cfg.size, height=0.0, center_zero=False)

        # Center the mesh (same as _get_terrain_mesh does)
        transform = np.eye(4)
        transform[0:2, -1] = -self.cfg.size[0] * 0.5, -self.cfg.size[1] * 0.5
        mesh.apply_transform(transform)

        # Origin at center
        origin = np.array([0.0, 0.0, 0.0])

        # Metadata marking this as a border (non-spawnable, steppable). Borders
        # are flat ground a robot can walk onto, so they advertise the same
        # locomotion skills a regular flat sub-terrain does — just not the ones
        # tied to elevation changes (no running on a thin strip). Sampling
        # probabilities must sum to 1.0.
        meta_data = {
            "needs_projection": False,  # Flat terrain - no projection needed
            "needs_directional_cmd": False,
            "is_border": True,  # Mark as border for spawning exclusion
            "skill_probs": {"walk_forward": 0.5, "standing": 0.5},
        }

        return mesh, origin, meta_data

    @override
    def _generate_random_terrains(self):
        """Add terrains based on randomly sampled difficulty parameter, with support for border columns."""
        # normalize the proportions of the sub-terrains
        proportions = np.array([sub_cfg.proportion for sub_cfg in self.cfg.sub_terrains.values()])
        proportions /= np.sum(proportions)
        # create a list of all terrain configs
        sub_terrains_cfgs = list(self.cfg.sub_terrains.values())

        # randomly sample sub-terrains
        for index in range(self.cfg.num_rows * self.cfg.num_cols):
            # coordinate index of the sub-terrain
            (sub_row, sub_col) = np.unravel_index(index, (self.cfg.num_rows, self.cfg.num_cols))

            if sub_col in self._border_columns:
                # Generate flat border terrain
                mesh, origin, meta_data = self._generate_flat_border_terrain()
                # Add border terrain without flat patch sampling
                self._add_sub_terrain(mesh, origin, meta_data, sub_row, sub_col, None)
            else:
                # randomly sample terrain index
                sub_index = self.np_rng.choice(len(proportions), p=proportions)
                # randomly sample difficulty parameter
                difficulty = self.np_rng.uniform(*self.cfg.difficulty_range)
                # generate terrain
                mesh, origin, meta_data = self._get_terrain_mesh(difficulty, sub_terrains_cfgs[sub_index])
                # add to sub-terrains
                self._add_sub_terrain(mesh, origin, meta_data, sub_row, sub_col, sub_terrains_cfgs[sub_index])

    @override
    def _generate_curriculum_terrains(self):
        """Add terrains based on the difficulty parameter, with support for border columns."""
        # normalize the proportions of the sub-terrains
        proportions = np.array([sub_cfg.proportion for sub_cfg in self.cfg.sub_terrains.values()])
        proportions /= np.sum(proportions)

        # find the sub-terrain index for each CONTENT column (not border columns)
        # we generate the terrains based on their proportion (not randomly sampled)
        sub_indices = []
        for index in range(self._content_num_cols):
            sub_index = np.min(np.where(index / self._content_num_cols + 0.001 < np.cumsum(proportions))[0])
            sub_indices.append(sub_index)
        sub_indices = np.array(sub_indices, dtype=np.int32)
        # create a list of all terrain configs
        sub_terrains_cfgs = list(self.cfg.sub_terrains.values())

        # curriculum-based sub-terrains with border handling
        content_col_idx = 0
        for sub_col in range(self.cfg.num_cols):
            for sub_row in range(self.cfg.num_rows):
                if sub_col in self._border_columns:
                    # Generate flat border terrain
                    mesh, origin, meta_data = self._generate_flat_border_terrain()
                    # Add border terrain without flat patch sampling
                    self._add_sub_terrain(mesh, origin, meta_data, sub_row, sub_col, None)
                else:
                    # Regular terrain generation
                    # vary the difficulty parameter linearly over the number of rows
                    # note: based on the proportion, multiple columns can have the same sub-terrain type.
                    #  Thus to increase the diversity along the rows, we add a small random value to the difficulty.
                    #  This ensures that the terrains are not exactly the same. For example, if the
                    #  the row index is 2 and the number of rows is 10, the nominal difficulty is 0.2.
                    #  We add a small random value to the difficulty to make it between 0.2 and 0.3.
                    lower, upper = self.cfg.difficulty_range
                    difficulty = (sub_row + self.np_rng.uniform()) / self.cfg.num_rows
                    difficulty = lower + (upper - lower) * difficulty
                    # generate terrain
                    mesh, origin, meta_data = self._get_terrain_mesh(difficulty, sub_terrains_cfgs[sub_indices[content_col_idx]])
                    # add to sub-terrains
                    self._add_sub_terrain(mesh, origin, meta_data, sub_row, sub_col, sub_terrains_cfgs[sub_indices[content_col_idx]])

            # Only increment content column index for non-border columns
            if sub_col not in self._border_columns:
                content_col_idx += 1

    @override
    def _add_sub_terrain(
        self, mesh: trimesh.Trimesh, origin: np.ndarray, meta_data: dict, row: int, col: int, sub_terrain_cfg: SubTerrainBaseCfg | None
    ):
        """Add input sub-terrain to the list of sub-terrains.

        This function adds the input sub-terrain mesh to the list of sub-terrains and updates the origin
        of the sub-terrain in the list of origins. It also samples flat patches if specified.

        Args:
            mesh: The mesh of the sub-terrain.
            origin: The origin of the sub-terrain.
            meta_data: The metadata of the sub-terrain.
            row: The row index of the sub-terrain.
            col: The column index of the sub-terrain.
            sub_terrain_cfg: The configuration of the sub-terrain, or None for border terrains.
        """
        if sub_terrain_cfg is not None:
            # Regular terrain - use parent's implementation
            super()._add_sub_terrain(mesh, origin, row, col, sub_terrain_cfg)
        else:
            # Border terrain - add mesh directly without flat patch sampling
            transform = np.eye(4)
            transform[0:2, -1] = (row + 0.5) * self.cfg.size[0], (col + 0.5) * self.cfg.size[1]
            mesh.apply_transform(transform)
            self.terrain_meshes.append(mesh)
            self.terrain_origins[row, col] = origin + transform[:3, -1]

        self.terrain_metadata[(row, col)] = meta_data

    @override
    def _get_terrain_mesh(self, difficulty: float, cfg: SubTerrainBaseCfg) -> tuple[trimesh.Trimesh, np.ndarray]:
        """Generate a sub-terrain mesh based on the input difficulty parameter.

        If caching is enabled, the sub-terrain is cached and loaded from the cache if it exists.
        The cache is stored in the cache directory specified in the configuration.

        .. Note:
            This function centers the 2D center of the mesh and its specified origin such that the
            2D center becomes :math:`(0, 0)` instead of :math:`(size[0] / 2, size[1] / 2).

        Args:
            difficulty: The difficulty parameter.
            cfg: The configuration of the sub-terrain.

        Returns:
            The sub-terrain mesh and origin.
        """
        # copy the configuration
        cfg = cfg.copy()
        # add other parameters to the sub-terrain configuration
        cfg.difficulty = float(difficulty)
        cfg.seed = self.cfg.seed
        # generate hash for the sub-terrain
        sub_terrain_hash = dict_to_md5_hash(cfg.to_dict())
        # generate the file name
        sub_terrain_cache_dir = os.path.join(self.cfg.cache_dir, sub_terrain_hash)
        sub_terrain_obj_filename = os.path.join(sub_terrain_cache_dir, "mesh.obj")
        sub_terrain_csv_filename = os.path.join(sub_terrain_cache_dir, "origin.csv")
        sub_terrain_meta_filename = os.path.join(sub_terrain_cache_dir, "cfg.yaml")

        # check if hash exists - if true, load the mesh and origin and return
        if self.cfg.use_cache and os.path.exists(sub_terrain_obj_filename):
            # load existing mesh
            mesh = trimesh.load_mesh(sub_terrain_obj_filename, process=False)
            origin = np.loadtxt(sub_terrain_csv_filename, delimiter=",")
            # return the generated mesh
            return mesh, origin

        # generate the terrain
        meshes, origin, meta_data = cfg.function(difficulty, cfg)
        mesh = trimesh.util.concatenate(meshes)
        # offset mesh such that they are in their center
        transform = np.eye(4)
        transform[0:2, -1] = -cfg.size[0] * 0.5, -cfg.size[1] * 0.5
        mesh.apply_transform(transform)
        # change origin to be in the center of the sub-terrain
        origin += transform[0:3, -1]

        # if caching is enabled, save the mesh and origin
        if self.cfg.use_cache:
            # create the cache directory
            os.makedirs(sub_terrain_cache_dir, exist_ok=True)
            # save the data
            mesh.export(sub_terrain_obj_filename)
            np.savetxt(sub_terrain_csv_filename, origin, delimiter=",", header="x,y,z")
            dump_yaml(sub_terrain_meta_filename, cfg)
        # return the generated mesh
        return mesh, origin, meta_data
