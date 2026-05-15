"""Pure-cfg companion to ``meta_composite_importer.py``.

Kept separate so importing this cfg does NOT pull in the runtime importer (and
its transitive ``isaaclab.markers`` chain that pre-loads ``pxr`` before
``AppLauncher`` runs). The runtime class is resolved via the string
``class_type`` when ``cfg.class_type(cfg)`` is called inside the sim.
"""

from isaaclab.utils import configclass

from .meta_terrain_importer_cfg import MetaTerrainImporterCfg


@configclass
class MetaCompositeTerrainImporterCfg(MetaTerrainImporterCfg):
    """Configuration for :class:`MetaCompositeTerrainImporter`."""

    class_type: type | str = "{DIR}.meta_composite_importer:MetaCompositeTerrainImporter"

    block_outline_bar_width: float = 0.05
    """Thickness (m) of the outline bars drawn around each block when
    ``debug_vis=True``."""

    block_outline_bar_height: float = 0.05
    """Vertical extent (m) of the outline bars above the cell origin when
    ``debug_vis=True``."""

    block_outline_z_offset: float = 0.02
    """Vertical lift (m) added on top of each cell origin so outline bars
    sit above the terrain surface."""

    block_outline_colors: dict[str, tuple[float, float, float]] | None = None
    """Optional override mapping block-type name -> RGB color in ``[0, 1]``.
    Unrecognised block types fall back to the importer's built-in palette,
    with a final grey fallback for anything still missing."""
