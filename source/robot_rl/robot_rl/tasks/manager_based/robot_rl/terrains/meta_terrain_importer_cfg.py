from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from .meta_terrain_importer import MetaTerrainImporter


@configclass
class MetaTerrainImporterCfg(TerrainImporterCfg):
    """Configuration for importing meta terrains."""

    class_type: type = MetaTerrainImporter
