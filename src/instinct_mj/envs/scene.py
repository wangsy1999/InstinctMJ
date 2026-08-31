from __future__ import annotations

from mjlab.scene import Scene
from mjlab.terrains import TerrainEntity

from instinct_mj.terrains import TerrainImporterCfg


class InstinctScene(Scene):
    """Scene variant that honors terrain cfg.class_type in Instinct tasks."""

    def _add_terrain(self) -> None:
        if self._cfg.terrain is None:
            return
        terrain_cfg = self._cfg.terrain
        terrain_cfg.num_envs = self.num_envs
        terrain_cfg.env_spacing = self.env_spacing
        if isinstance(terrain_cfg, TerrainImporterCfg):
            terrain = terrain_cfg.class_type(terrain_cfg, device=self.device)
        else:
            terrain = TerrainEntity(terrain_cfg, device=self.device)
        self._terrain = terrain
        self.entities["terrain"] = terrain
        frame = self.spec.worldbody.add_frame()
        self.spec.attach(terrain.spec, prefix="", frame=frame)
