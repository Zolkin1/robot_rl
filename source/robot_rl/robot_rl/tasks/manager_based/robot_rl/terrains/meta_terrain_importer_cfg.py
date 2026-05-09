from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg


@configclass
class MetaTerrainImporterCfg(TerrainImporterCfg):
    """Configuration for importing meta terrains.

    ``class_type`` is a string so importing this cfg does NOT pull in the
    runtime ``MetaTerrainImporter`` (and its transitive ``isaaclab.markers``
    chain that pre-loads ``pxr``). The runtime class is resolved when
    ``cfg.class_type(cfg)`` is called inside the sim, after AppLauncher.
    """

    class_type: type | str = "{DIR}.meta_terrain_importer:MetaTerrainImporter"

    skill_list: list[str] = ["standing"]