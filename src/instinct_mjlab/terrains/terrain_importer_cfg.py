from __future__ import annotations
from dataclasses import dataclass, field

from typing import Literal

from mjlab.terrains import TerrainImporterCfg as TerrainImporterCfgBase

from .terrain_importer import TerrainImporter
from .virtual_obstacle import VirtualObstacleCfg


@dataclass(kw_only=True)
class TerrainImporterCfg(TerrainImporterCfgBase):
    class_type: type = TerrainImporter
    """The inherited class to use for the terrain importer."""

    virtual_obstacles: dict[str, VirtualObstacleCfg] = field(default_factory=dict)
    """The virtual obstacles to use for the terrain importer."""

    virtual_obstacle_source: Literal["mesh", "heightfield"] = "mesh"
    """Source used to generate virtual obstacles.

    - ``"mesh"``: Use concatenated terrain mesh surface.
    - ``"heightfield"``: Extract obstacle edges directly from MuJoCo hfield data.
    """

    virtual_obstacle_hfield_height_threshold: float | None = 0.04
    """Absolute height-difference threshold (meters) used by hfield edge extraction.

    If set to ``None``, the extractor falls back to angle-based conversion:
    ``tan(angle_threshold) * hfield_cell_size``.
    """

    virtual_obstacle_hfield_merge_runs: bool = True
    """Whether to merge contiguous hfield edge intervals into long line segments."""

    virtual_obstacle_hfield_project_to_high_side: bool = True
    """Whether to project hfield edge segments to the higher side (convex crest side)."""

    collision_debug_vis: bool = False
    """Whether to visualize terrain collision geoms by tinting them in purple."""

    collision_debug_rgba: tuple[float, float, float, float] = (0.62, 0.2, 0.9, 0.35)
    """RGBA tint for terrain collision debug visualization."""

    terrain_type: Literal["generator", "plane", "hacked_generator"] = "generator"
    """The type of terrain to generate. Defaults to "generator".

    Available options are "plane" and "generator".

    ## NOTE
    The TerrainImporter keeps "hacked_generator" as a compatibility alias.
    Runtime behavior is mjlab-native and routes "hacked_generator" through
    the same generator pipeline as "generator".
    """
