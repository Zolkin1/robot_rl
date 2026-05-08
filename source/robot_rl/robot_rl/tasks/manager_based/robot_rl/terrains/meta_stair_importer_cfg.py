"""Pure-cfg companion to ``meta_stair_importer.py``.

Kept separate so importing this cfg does NOT pull in the runtime importer (and
its transitive ``isaaclab.markers`` chain that pre-loads ``pxr`` before
``AppLauncher`` runs). The runtime class is resolved via the string
``class_type`` when ``cfg.class_type(cfg)`` is called inside the sim.
"""

from isaaclab.utils import configclass

from .meta_terrain_importer_cfg import MetaTerrainImporterCfg


@configclass
class MetaStairTerrainImporterCfg(MetaTerrainImporterCfg):
    """Configuration for :class:`MetaStairTerrainImporter`."""

    class_type: type | str = "{DIR}.meta_stair_importer:MetaStairTerrainImporter"
