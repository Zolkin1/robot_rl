"""Composable terrain blocks for assembling heterogeneous sub-terrains.

A :class:`CompositeSubTerrainCfg` stacks an ordered list of
:class:`TerrainBlockCfg` instances along the sub-terrain's x-axis. Each block
owns its mesh footprint, its skill distribution, and any block-specific
metadata (e.g. stair-tread centers). At runtime the per-block metadata is
exposed by :class:`MetaCompositeTerrainImporter`, which lets the MDP query
``skill_probs_at(xy)`` and project to stair tops on a per-block resolution
within a single sub-terrain cell.
"""

from .base import (
    BlockOutput,
    TerrainBlock,
    TerrainBlockCfg,
    validate_skill_probs,
)
from .composite import CompositeSubTerrainCfg, composite_terrain
from .flat import FlatBlock, FlatBlockCfg
from .randomized_composite import (
    BlockChoice,
    RandomizedCompositeSubTerrainCfg,
    randomized_composite_terrain,
)
from .slope import SlopeBlock, SlopeBlockCfg
from .stair import StairBlock, StairBlockCfg

__all__ = [
    "BlockChoice",
    "BlockOutput",
    "CompositeSubTerrainCfg",
    "FlatBlock",
    "FlatBlockCfg",
    "RandomizedCompositeSubTerrainCfg",
    "SlopeBlock",
    "SlopeBlockCfg",
    "StairBlock",
    "StairBlockCfg",
    "TerrainBlock",
    "TerrainBlockCfg",
    "composite_terrain",
    "randomized_composite_terrain",
    "validate_skill_probs",
]
