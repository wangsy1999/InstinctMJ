from __future__ import annotations

import copy
import math
import numpy as np
import torch
import trimesh
import time
from typing import TYPE_CHECKING

import mujoco
from mjlab.terrains import SubTerrainCfg as SubTerrainBaseCfg
from mjlab.terrains import TerrainGenerator
from mjlab.terrains import TerrainImporter as TerrainImporterBase


class Timer:
    """Simple timer context manager."""

    def __init__(self, message: str):
        self.message = message
        self._start = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        elapsed = time.perf_counter() - self._start
        print(f"{self.message}: {elapsed:.4f}s")
        return False

if TYPE_CHECKING:
    from .terrain_importer_cfg import TerrainImporterCfg
    from .virtual_obstacle import VirtualObstacleBase


class TerrainImporter(TerrainImporterBase):
    def __init__(self, cfg: TerrainImporterCfg, device: str):
        runtime_terrain_type = cfg.terrain_type
        if runtime_terrain_type == "hacked_generator":
            # Keep the public name for compatibility, but route to mjlab native generator pipeline.
            runtime_terrain_type = "generator"

        self._debug_vis_enabled = False
        self._collision_debug_vis = bool(getattr(cfg, "collision_debug_vis", False))
        self._collision_debug_rgba = tuple(getattr(cfg, "collision_debug_rgba", (0.62, 0.2, 0.9, 0.35)))
        self._virtual_obstacle_source = str(getattr(cfg, "virtual_obstacle_source", "mesh")).lower()
        if self._virtual_obstacle_source not in ("mesh", "heightfield"):
            raise ValueError(
                "virtual_obstacle_source must be 'mesh' or 'heightfield'. "
                f"Got: {self._virtual_obstacle_source!r}"
            )
        self._virtual_obstacles = {}
        for name, virtual_obstacle_cfg in cfg.virtual_obstacles.items():
            if virtual_obstacle_cfg is None:
                continue
            virtual_obstacle = virtual_obstacle_cfg.class_type(virtual_obstacle_cfg)
            self._virtual_obstacles[name] = virtual_obstacle

        # Build a runtime cfg that keeps all fields but maps hacked_generator -> generator.
        cfg_runtime = copy.deepcopy(cfg)
        cfg_runtime.terrain_type = runtime_terrain_type

        # -----------------------------------------------------------------------------
        # mjlab-native terrain importer flow (mirrors mjlab.terrains.TerrainImporter)
        # -----------------------------------------------------------------------------
        self.cfg = cfg_runtime
        self.device = device
        self._spec = mujoco.MjSpec()
        self.env_origins = None
        self.terrain_origins = None
        self.terrain_generator = None

        if self.cfg.terrain_type == "generator":
            if self.cfg.terrain_generator is None:
                raise ValueError(
                    "Input terrain type is 'generator' but no value provided for 'terrain_generator'."
                )
            terrain_generator_cls = getattr(
                self.cfg.terrain_generator,
                "class_type",
                TerrainGenerator,
            )
            self.terrain_generator = terrain_generator_cls(
                self.cfg.terrain_generator,
                device=self.device,
            )
            self.terrain_generator.compile(self._spec)
            self.configure_env_origins(self.terrain_generator.terrain_origins)
            self._flat_patches = {
                name: torch.from_numpy(arr).to(device=self.device, dtype=torch.float)
                for name, arr in self.terrain_generator.flat_patches.items()
            }
            self._flat_patch_radii = dict(self.terrain_generator.flat_patch_radii)
        elif self.cfg.terrain_type == "plane":
            self.import_ground_plane("terrain")
            self.configure_env_origins()
            self._flat_patches = {}
            self._flat_patch_radii = {}
        else:
            raise ValueError(f"Unknown terrain type: {self.cfg.terrain_type}")

        self._add_env_origin_sites()
        self._add_terrain_origin_sites()
        self._add_flat_patch_sites()

        if len(self._virtual_obstacles) > 0:
            generated_from_hfield = False
            if self._virtual_obstacle_source == "heightfield":
                generated_from_hfield = self._generate_virtual_obstacles_from_heightfield()

            if not generated_from_hfield:
                terrain_mesh = self._get_terrain_mesh_for_virtual_obstacles()
                if terrain_mesh is None:
                    raise RuntimeError(
                        "virtual obstacles are configured but no terrain mesh is available from terrain generator."
                    )
                self._generate_virtual_obstacles(terrain_mesh)
            self._release_terrain_mesh_cache()

        if self._collision_debug_vis:
            self._apply_collision_debug_visual_style()

    @property
    def virtual_obstacles(self) -> dict[str, VirtualObstacleBase]:
        """Get the virtual obstacles representing the edges.
        TODO: Make the returned value more general.
        """
        # still pointing the same VirtualObstacleBase objects but the dict is a copy.
        return self._virtual_obstacles.copy()

    @property
    def subterrain_specific_cfgs(self) -> list[SubTerrainBaseCfg] | None:
        """Get the specific configurations for all subterrains."""
        # This is a placeholder. The actual implementation should return the specific configurations.
        return (
            self.terrain_generator.subterrain_specific_cfgs
            if hasattr(self, "terrain_generator") and hasattr(self.terrain_generator, "subterrain_specific_cfgs")
            else None
        )

    """
    Operations - Import.
    """

    def import_ground_plane(self, name: str):
        """Import ground plane via mjlab base implementation."""
        return super().import_ground_plane(name)

    def _get_terrain_mesh_for_virtual_obstacles(self) -> trimesh.Trimesh | None:
        if self.terrain_generator is None:
            return None
        terrain_mesh = getattr(self.terrain_generator, "terrain_mesh", None)
        return terrain_mesh

    def _generate_virtual_obstacles(self, mesh: trimesh.Trimesh):
        """Generate virtual obstacles from a terrain mesh."""
        mesh.merge_vertices()
        mesh.update_faces(mesh.unique_faces())  # remove duplicate faces
        mesh.remove_unreferenced_vertices()
        # Generate virtual obstacles based on the generated terrain mesh.
        # NOTE: generate virtual obstacle first because it might modify the mesh.
        for name, virtual_obstacle in self._virtual_obstacles.items():
            with Timer(f"Generate virtual obstacle {name}"):
                virtual_obstacle.generate(mesh, device=self.device)

    @staticmethod
    def _iter_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
        """Return inclusive index ranges of contiguous True runs."""
        runs: list[tuple[int, int]] = []
        idx = 0
        while idx < mask.size:
            if not bool(mask[idx]):
                idx += 1
                continue
            start = idx
            while idx + 1 < mask.size and bool(mask[idx + 1]):
                idx += 1
            runs.append((start, idx))
            idx += 1
        return runs

    @staticmethod
    def _active_interval_runs_with_vertex_split(
        interval_active: np.ndarray,
        vertex_coord: np.ndarray,
        enforce_vertex_coord_continuity: bool,
    ) -> list[tuple[int, int]]:
        """Return active interval runs, optionally split when projected vertex coord changes."""
        if not enforce_vertex_coord_continuity:
            return TerrainImporter._iter_true_runs(interval_active)

        runs: list[tuple[int, int]] = []
        start: int | None = None
        for interval_idx, is_active in enumerate(interval_active):
            if not bool(is_active):
                if start is not None:
                    runs.append((start, interval_idx - 1))
                    start = None
                continue
            if start is None:
                start = interval_idx
                continue
            # Split run when the projected edge switches side at shared vertex.
            if not np.isclose(vertex_coord[interval_idx], vertex_coord[interval_idx - 1]):
                runs.append((start, interval_idx - 1))
                start = interval_idx
        if start is not None:
            runs.append((start, interval_active.size - 1))
        return runs

    @staticmethod
    def _extract_edge_segments_from_hfield(
        hfield_spec,
        geom_pos: np.ndarray,
        angle_threshold: float,
        height_threshold: float | None,
        merge_runs: bool,
        project_to_high_side: bool,
    ) -> np.ndarray:
        """Extract edge line segments directly from one MuJoCo hfield surface."""
        nrow = int(getattr(hfield_spec, "nrow", 0))
        ncol = int(getattr(hfield_spec, "ncol", 0))
        if nrow <= 1 or ncol <= 1:
            return np.empty((0, 6), dtype=np.float32)

        userdata = np.asarray(getattr(hfield_spec, "userdata", []), dtype=np.float64)
        if userdata.size != nrow * ncol:
            return np.empty((0, 6), dtype=np.float32)
        normalized_height = userdata.reshape(nrow, ncol)

        size = np.asarray(getattr(hfield_spec, "size", []), dtype=np.float64).reshape(-1)
        if size.size < 4:
            return np.empty((0, 6), dtype=np.float32)
        half_x, half_y, elevation_range, _base_thickness = size[:4]

        x_coords = np.linspace(float(geom_pos[0] - half_x), float(geom_pos[0] + half_x), nrow, dtype=np.float64)
        y_coords = np.linspace(float(geom_pos[1] - half_y), float(geom_pos[1] + half_y), ncol, dtype=np.float64)
        z_coords = float(geom_pos[2]) + normalized_height * float(elevation_range)

        if height_threshold is None:
            dx = abs(x_coords[1] - x_coords[0]) if nrow > 1 else 0.0
            dy = abs(y_coords[1] - y_coords[0]) if ncol > 1 else 0.0
            positive_steps = [v for v in (dx, dy) if v > 0.0]
            base_step = max(min(positive_steps), 1.0e-6) if len(positive_steps) > 0 else 1.0e-6
            dz_threshold = math.tan(math.radians(float(angle_threshold))) * base_step
        else:
            dz_threshold = float(height_threshold)
        dz_threshold = max(dz_threshold, 1.0e-6)

        segments: list[list[float]] = []

        # Edges along x-boundaries (vertical lines in XY plane).
        for i in range(nrow - 1):
            z0 = z_coords[i, :]
            z1 = z_coords[i + 1, :]
            dz_col = np.abs(z1 - z0)
            interval_active = (dz_col[:-1] > dz_threshold) | (dz_col[1:] > dz_threshold)
            if not np.any(interval_active):
                continue
            z_line = np.maximum(z0, z1)
            if project_to_high_side:
                x_line = np.where(z0 >= z1, x_coords[i], x_coords[i + 1])
            else:
                x_line = np.full(ncol, 0.5 * (x_coords[i] + x_coords[i + 1]), dtype=np.float64)

            if merge_runs:
                runs = TerrainImporter._active_interval_runs_with_vertex_split(
                    interval_active=interval_active,
                    vertex_coord=x_line,
                    enforce_vertex_coord_continuity=project_to_high_side,
                )
                for start, end in runs:
                    end_vertex = end + 1
                    segments.append(
                        [x_line[start], y_coords[start], z_line[start], x_line[end_vertex], y_coords[end_vertex], z_line[end_vertex]]
                    )
            else:
                for j in range(ncol - 1):
                    if not interval_active[j]:
                        continue
                    segments.append([x_line[j], y_coords[j], z_line[j], x_line[j + 1], y_coords[j + 1], z_line[j + 1]])

        # Edges along y-boundaries (horizontal lines in XY plane).
        for j in range(ncol - 1):
            z0 = z_coords[:, j]
            z1 = z_coords[:, j + 1]
            dz_row = np.abs(z1 - z0)
            interval_active = (dz_row[:-1] > dz_threshold) | (dz_row[1:] > dz_threshold)
            if not np.any(interval_active):
                continue
            z_line = np.maximum(z0, z1)
            if project_to_high_side:
                y_line = np.where(z0 >= z1, y_coords[j], y_coords[j + 1])
            else:
                y_line = np.full(nrow, 0.5 * (y_coords[j] + y_coords[j + 1]), dtype=np.float64)

            if merge_runs:
                runs = TerrainImporter._active_interval_runs_with_vertex_split(
                    interval_active=interval_active,
                    vertex_coord=y_line,
                    enforce_vertex_coord_continuity=project_to_high_side,
                )
                for start, end in runs:
                    end_vertex = end + 1
                    segments.append(
                        [x_coords[start], y_line[start], z_line[start], x_coords[end_vertex], y_line[end_vertex], z_line[end_vertex]]
                    )
            else:
                for i in range(nrow - 1):
                    if not interval_active[i]:
                        continue
                    segments.append([x_coords[i], y_line[i], z_line[i], x_coords[i + 1], y_line[i + 1], z_line[i + 1]])

        if len(segments) == 0:
            return np.empty((0, 6), dtype=np.float32)
        return np.asarray(segments, dtype=np.float32)

    def _collect_hfield_edge_segments(
        self,
        angle_threshold: float,
        height_threshold: float | None,
        merge_runs: bool,
        project_to_high_side: bool,
    ) -> np.ndarray:
        """Collect edge segments from all terrain hfields in the scene spec."""
        terrain_body = self._spec.body("terrain")
        if terrain_body is None:
            return np.empty((0, 6), dtype=np.float32)

        all_segments: list[np.ndarray] = []
        for geom in terrain_body.geoms:
            hfield_name = getattr(geom, "hfieldname", "")
            if not isinstance(hfield_name, str) or hfield_name == "":
                continue
            hfield_spec = self._spec.hfield(hfield_name)
            if hfield_spec is None:
                continue
            geom_pos = np.asarray(getattr(geom, "pos", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
            segments = self._extract_edge_segments_from_hfield(
                hfield_spec=hfield_spec,
                geom_pos=geom_pos,
                angle_threshold=angle_threshold,
                height_threshold=height_threshold,
                merge_runs=merge_runs,
                project_to_high_side=project_to_high_side,
            )
            if segments.size > 0:
                all_segments.append(segments)

        if len(all_segments) == 0:
            return np.empty((0, 6), dtype=np.float32)
        return np.concatenate(all_segments, axis=0).astype(np.float32, copy=False)

    def _generate_virtual_obstacles_from_heightfield(self) -> bool:
        """Generate virtual obstacles directly from MuJoCo hfield data."""
        if len(self._virtual_obstacles) == 0:
            return True

        configured_height_threshold = getattr(self.cfg, "virtual_obstacle_hfield_height_threshold", 0.04)
        if configured_height_threshold is not None:
            configured_height_threshold = float(configured_height_threshold)
            if configured_height_threshold <= 0.0:
                configured_height_threshold = None
        merge_runs = bool(getattr(self.cfg, "virtual_obstacle_hfield_merge_runs", True))
        project_to_high_side = bool(getattr(self.cfg, "virtual_obstacle_hfield_project_to_high_side", True))

        generated_any = False
        for name, virtual_obstacle in self._virtual_obstacles.items():
            if not hasattr(virtual_obstacle, "generate_from_edge_segments"):
                print(
                    f"[WARNING] virtual obstacle '{name}' does not support heightfield source; "
                    "falling back to mesh source."
                )
                return False

            angle_threshold = float(getattr(virtual_obstacle, "angle_threshold", 70.0))
            edge_segments = self._collect_hfield_edge_segments(
                angle_threshold=angle_threshold,
                height_threshold=configured_height_threshold,
                merge_runs=merge_runs,
                project_to_high_side=project_to_high_side,
            )
            with Timer(f"Generate virtual obstacle {name} from heightfield"):
                virtual_obstacle.generate_from_edge_segments(edge_segments, device=self.device)
            generated_any = True
        return generated_any

    def _release_terrain_mesh_cache(self) -> None:
        """Release terrain mesh cache once virtual obstacles are built."""
        if self.terrain_generator is None:
            return
        if hasattr(self.terrain_generator, "terrain_mesh"):
            self.terrain_generator.terrain_mesh = None
        terrain_meshes = getattr(self.terrain_generator, "_terrain_meshes", None)
        if isinstance(terrain_meshes, list):
            terrain_meshes.clear()

    def _apply_collision_debug_visual_style(self) -> None:
        """Tint terrain collision geoms so they are visible in the viewer."""
        terrain_body = self._spec.body("terrain")
        if terrain_body is None:
            return
        for geom in terrain_body.geoms:
            contype = int(getattr(geom, "contype", 1))
            conaffinity = int(getattr(geom, "conaffinity", 1))
            if contype == 0 and conaffinity == 0:
                continue
            geom.rgba[:] = self._collision_debug_rgba

    def set_debug_vis(self, debug_vis: bool) -> bool:
        """Set the debug visualization flag.

        Args:
            vis: True to enable debug visualization, False to disable.
        """
        self._debug_vis_enabled = debug_vis
        results = True

        for name, virtual_obstacle in self._virtual_obstacles.items():
            if debug_vis:
                virtual_obstacle.visualize()
            else:
                virtual_obstacle.disable_visualizer()

        return results

    def debug_vis(self, visualizer) -> None:
        """Draw virtual obstacles using viewer-native debug visualizer."""
        if not self._debug_vis_enabled:
            return
        for virtual_obstacle in self._virtual_obstacles.values():
            if hasattr(virtual_obstacle, "debug_vis"):
                virtual_obstacle.debug_vis(visualizer)

    def configure_env_origins(self, origins: np.ndarray | torch.Tensor | None = None):
        """Configure the environment origins.

        Args:
            origins: The origins of the environments. Shape is (num_envs, 3).
        """
        return super().configure_env_origins(origins)
