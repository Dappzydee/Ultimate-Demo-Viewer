"""Persistent normalized demo and map state for the integrated viewer app."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
import json
from pathlib import Path
from typing import Any
import zipfile

import numpy as np
import trimesh
from awpy import Demo

from .analysis import VisibilityAnalyzer, resolve_tri_path
from .flash_coverage import FlashCoverageAnalyzer, FlashCoverageConfig, FlashCoverageResult
from .flash_events import FlashDetonation, _first_value
from .geometry import export_glb_bytes, interior_sample_points, load_tri_mesh
from .models import AnalysisConfig, PlayerPose, VisibilityTimelineResult
from .raycasting import create_raycaster


SESSION_SCHEMA_VERSION = 1
ROUND_EVENTS = ["round_start", "round_freeze_end", "round_officially_ended", "flashbang_detonate"]


@dataclass(frozen=True)
class RoundInfo:
    number: int
    start_tick: int
    freeze_end_tick: int
    end_tick: int


@dataclass(frozen=True)
class PlayerInfo:
    id: str
    name: str
    steamid: str | None


class MapAnalysisContext:
    """Cache geometry-derived samples and raycasters for repeated analyses."""

    def __init__(self, mesh: trimesh.Trimesh) -> None:
        self.mesh = mesh
        self._samples: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._raycasters: dict[bool, object] = {}

    def samples(self, count: int) -> tuple[np.ndarray, np.ndarray]:
        if count not in self._samples:
            self._samples[count] = interior_sample_points(self.mesh, count)
        return self._samples[count]

    def raycaster(self, prefer_gpu: bool) -> object:
        if prefer_gpu not in self._raycasters:
            self._raycasters[prefer_gpu] = create_raycaster(self.mesh, prefer_gpu)
        return self._raycasters[prefer_gpu]

    def visibility_analyzer(self, config: AnalysisConfig) -> VisibilityAnalyzer:
        points, faces = self.samples(config.samples_per_triangle)
        return VisibilityAnalyzer(
            self.mesh, config, sample_points=points, sample_face_ids=faces,
            raycaster=self.raycaster(config.prefer_gpu),
        )

    def flash_analyzer(self, config: FlashCoverageConfig) -> FlashCoverageAnalyzer:
        points, faces = self.samples(config.samples_per_triangle)
        return FlashCoverageAnalyzer(
            self.mesh, config, sample_points=points, sample_face_ids=faces,
            raycaster=self.raycaster(config.prefer_gpu),
        )

    def pick_surface(self, origin: np.ndarray, direction: np.ndarray) -> np.ndarray | None:
        """Return the nearest map intersection for one UI placement ray."""
        direction = np.asarray(direction, dtype=np.float64)
        length = np.linalg.norm(direction)
        if length <= 1e-9:
            raise ValueError("Pick direction cannot be zero.")
        direction /= length
        origin = np.asarray(origin, dtype=np.float64)
        triangles = self.mesh.triangles
        edge1 = triangles[:, 1] - triangles[:, 0]
        edge2 = triangles[:, 2] - triangles[:, 0]
        h = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
        determinant = np.einsum("ij,ij->i", edge1, h)
        valid = np.abs(determinant) > 1e-9
        inverse = np.divide(1.0, determinant, out=np.zeros_like(determinant), where=valid)
        s = origin - triangles[:, 0]
        u = inverse * np.einsum("ij,ij->i", s, h)
        valid &= (u >= 0.0) & (u <= 1.0)
        q = np.cross(s, edge1)
        v = inverse * (q @ direction)
        valid &= (v >= 0.0) & ((u + v) <= 1.0)
        distances = inverse * np.einsum("ij,ij->i", edge2, q)
        valid &= distances > 1e-5
        if not np.any(valid):
            return None
        distance = float(np.min(distances[valid]))
        return origin + direction * distance


class DemoSession:
    """A demo parsed once and reusable for any number of analyses."""

    def __init__(
        self, *, source_name: str, map_name: str, tick_rate: float,
        rounds: list[RoundInfo], players: list[PlayerInfo],
        round_players: dict[tuple[int, int], dict[str, str | None]],
        pose_ticks: np.ndarray, pose_rounds: np.ndarray, pose_players: np.ndarray,
        pose_positions: np.ndarray, pose_yaws: np.ndarray, pose_pitches: np.ndarray,
        pose_ducks: np.ndarray, flashes: list[FlashDetonation], mesh: trimesh.Trimesh,
    ) -> None:
        self.source_name = source_name
        self.map_name = map_name
        self.tick_rate = float(tick_rate)
        self.rounds = rounds
        self.players = players
        self.round_players = round_players
        self.pose_ticks = pose_ticks
        self.pose_rounds = pose_rounds
        self.pose_players = pose_players
        self.pose_positions = pose_positions
        self.pose_yaws = pose_yaws
        self.pose_pitches = pose_pitches
        self.pose_ducks = pose_ducks
        self.flashes = flashes
        self.mesh = mesh
        self.map_context = MapAnalysisContext(mesh)
        self._player_index = {player.id: index for index, player in enumerate(players)}
        self._round_index = {round_info.number: round_info for round_info in rounds}
        self._geometry_glb: bytes | None = None
        self.saved_analysis_metadata: dict[str, Any] | None = None
        self.saved_analysis_data: bytes | None = None

    @classmethod
    def from_demo(
        cls, demo_path: Path, *, tri_path: Path | None = None,
        tick_rate_override: float | None = None,
    ) -> "DemoSession":
        demo = Demo(str(demo_path), verbose=False)
        demo.parse(
            player_props=["X", "Y", "Z", "pitch", "yaw", "duck_amount", "team_name", "team_clan_name"],
            events=ROUND_EVENTS,
        )
        map_name = demo.header.get("map_name") or demo.header.get("map")
        if not map_name:
            raise ValueError("Could not determine the map from the demo.")
        header_rate = demo.header.get("tick_rate")
        if tick_rate_override is not None and tick_rate_override <= 0:
            raise ValueError("Tick rate override must be positive.")
        tick_rate = tick_rate_override if tick_rate_override is not None else (
            float(header_rate) if isinstance(header_rate, (int, float)) and header_rate > 0 else 64.0
        )
        rounds = cls._normalize_rounds(demo.rounds)
        if not rounds:
            raise ValueError("The demo contains no usable rounds.")
        (
            players, round_players, pose_ticks, pose_rounds, pose_players,
            pose_positions, pose_yaws, pose_pitches, pose_ducks,
        ) = cls._normalize_poses(demo.ticks)
        flashes = cls._normalize_flashes(
            demo.events.get("flashbang_detonate"), str(map_name), rounds,
            players, round_players, pose_ticks, pose_players,
        )
        mesh = load_tri_mesh(resolve_tri_path(str(map_name), tri_path))
        return cls(
            source_name=demo_path.name, map_name=str(map_name), tick_rate=tick_rate,
            rounds=rounds, players=players, round_players=round_players,
            pose_ticks=pose_ticks, pose_rounds=pose_rounds, pose_players=pose_players,
            pose_positions=pose_positions, pose_yaws=pose_yaws, pose_pitches=pose_pitches,
            pose_ducks=pose_ducks, flashes=flashes, mesh=mesh,
        )

    @staticmethod
    def _normalize_rounds(frame: Any) -> list[RoundInfo]:
        result = []
        for row in frame.sort("round_num").iter_rows(named=True):
            start = int(row["start"])
            freeze_end = int(row.get("freeze_end") if row.get("freeze_end") is not None else start)
            end_value = row.get("end") if row.get("end") is not None else row.get("official_end")
            end = int(end_value if end_value is not None else freeze_end)
            result.append(RoundInfo(int(row["round_num"]), start, freeze_end, end))
        return result

    @staticmethod
    def _normalize_poses(frame: Any) -> tuple[Any, ...]:
        required = {"tick", "round_num", "name", "X", "Y", "Z", "yaw", "pitch"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Demo ticks are missing required fields: {sorted(missing)}")
        players: list[PlayerInfo] = []
        player_indexes: dict[str, int] = {}
        round_players: dict[tuple[int, int], dict[str, str | None]] = {}
        ticks: list[int] = []
        rounds: list[int] = []
        pose_players: list[int] = []
        positions: list[tuple[float, float, float]] = []
        yaws: list[float] = []
        pitches: list[float] = []
        ducks: list[float] = []
        for row in frame.sort("tick").iter_rows(named=True):
            if any(row.get(name) is None for name in ("tick", "round_num", "name", "X", "Y", "Z", "yaw", "pitch")):
                continue
            steam_value = row.get("steamid")
            steamid = str(steam_value) if steam_value not in (None, 0, "0") else None
            player_id = f"steam:{steamid}" if steamid else f"name:{row['name']}"
            if player_id not in player_indexes:
                player_indexes[player_id] = len(players)
                players.append(PlayerInfo(player_id, str(row["name"]), steamid))
            player_index = player_indexes[player_id]
            round_number = int(row["round_num"])
            side = row.get("side") or row.get("team_name")
            team = row.get("team_clan_name")
            round_players[(round_number, player_index)] = {
                "side": str(side) if side else None,
                "team": str(team) if team else None,
            }
            ticks.append(int(row["tick"]))
            rounds.append(round_number)
            pose_players.append(player_index)
            positions.append((float(row["X"]), float(row["Y"]), float(row["Z"])))
            yaws.append(float(row["yaw"]))
            pitches.append(float(row["pitch"]))
            duck = row.get("duck_amount")
            ducks.append(float(duck) if duck is not None and np.isfinite(float(duck)) else 0.0)
        if not ticks:
            raise ValueError("The demo contains no usable player poses.")
        return (
            players, round_players, np.asarray(ticks, dtype=np.int64), np.asarray(rounds, dtype=np.int16),
            np.asarray(pose_players, dtype=np.int16), np.asarray(positions, dtype=np.float32),
            np.asarray(yaws, dtype=np.float32), np.asarray(pitches, dtype=np.float32),
            np.asarray(ducks, dtype=np.float32),
        )

    @classmethod
    def _normalize_flashes(
        cls, frame: Any, map_name: str, rounds: list[RoundInfo], players: list[PlayerInfo],
        round_players: dict[tuple[int, int], dict[str, str | None]],
        pose_ticks: np.ndarray, pose_players: np.ndarray,
    ) -> list[FlashDetonation]:
        if frame is None or frame.is_empty():
            return []
        player_by_steam = {player.steamid: index for index, player in enumerate(players) if player.steamid}
        player_by_name = {player.name: index for index, player in enumerate(players)}
        result = []
        for index, row in enumerate(frame.sort("tick").iter_rows(named=True)):
            tick = _first_value(row, "tick")
            x = _first_value(row, "X", "x", "user_X", "user_x")
            y = _first_value(row, "Y", "y", "user_Y", "user_y")
            z = _first_value(row, "Z", "z", "user_Z", "user_z")
            if tick is None or x is None or y is None or z is None:
                continue
            tick = int(tick)
            round_number = next(
                (item.number for item in rounds if item.start_tick <= tick <= item.end_tick), None,
            )
            thrower = _first_value(row, "thrower", "attacker_name", "user_name", "name")
            steam_value = _first_value(row, "thrower_steamid", "attacker_steamid", "user_steamid", "steamid")
            steamid = str(steam_value) if steam_value not in (None, 0, "0") else None
            side = _first_value(row, "thrower_side", "attacker_side", "user_side", "side")
            team = _first_value(
                row, "thrower_team_clan_name", "attacker_team_clan_name", "user_team_clan_name", "team_clan_name",
            )
            player_index = player_by_steam.get(steamid) if steamid else player_by_name.get(str(thrower))
            if player_index is not None and round_number is not None:
                assignment = round_players.get((round_number, player_index), {})
                side = side or assignment.get("side")
                team = team or assignment.get("team")
            result.append(FlashDetonation(
                index=index, tick=tick, position=(float(x), float(y), float(z)), map_name=map_name,
                round_number=round_number, thrower=str(thrower) if thrower else None,
                thrower_steamid=steamid, thrower_side=str(side) if side else None,
                thrower_team=str(team) if team else None,
            ))
        return result

    def metadata(self) -> dict[str, Any]:
        rounds = []
        for round_info in self.rounds:
            round_player_values = []
            for (number, player_index), assignment in self.round_players.items():
                if number != round_info.number:
                    continue
                player = self.players[player_index]
                round_player_values.append({
                    "id": player.id, "name": player.name, "steamid": player.steamid,
                    "side": assignment.get("side"), "team": assignment.get("team"),
                })
            round_player_values.sort(key=lambda item: ((item["side"] or ""), item["name"].lower()))
            rounds.append({
                "number": round_info.number,
                "startTick": round_info.start_tick,
                "freezeEndTick": round_info.freeze_end_tick,
                "endTick": round_info.end_tick,
                "durationSeconds": max(0.0, (round_info.end_tick - round_info.freeze_end_tick) / self.tick_rate),
                "players": round_player_values,
            })
        return {
            "sourceName": self.source_name,
            "mapName": self.map_name,
            "tickRate": self.tick_rate,
            "faceCount": len(self.mesh.faces),
            "vertexCount": len(self.mesh.vertices),
            "rounds": rounds,
            "flashes": [flash.to_json_dict() for flash in self.flashes],
        }

    def select_poses(
        self, player_id: str, round_number: int, start_seconds: float, end_seconds: float,
        *, instant: bool, tick_step: int, eye_height: float, crouch_eye_height: float,
    ) -> list[PlayerPose]:
        if player_id not in self._player_index:
            raise ValueError("Selected player is not present in this session.")
        if round_number not in self._round_index:
            raise ValueError(f"Round {round_number} does not exist in this session.")
        if start_seconds < 0 or end_seconds < start_seconds:
            raise ValueError("The selected time window is invalid.")
        if tick_step < 1:
            raise ValueError("Tick step must be at least 1.")
        player_index = self._player_index[player_id]
        round_info = self._round_index[round_number]
        duration_seconds = max(0.0, (round_info.end_tick - round_info.freeze_end_tick) / self.tick_rate)
        if start_seconds > duration_seconds or end_seconds > duration_seconds:
            raise ValueError(f"Selected time is outside round {round_number} (0-{duration_seconds:.1f} seconds).")
        base = (self.pose_players == player_index) & (self.pose_rounds == round_number)
        candidate_ids = np.flatnonzero(base)
        if not len(candidate_ids):
            raise ValueError("The selected player has no poses in this round.")
        lower = round_info.freeze_end_tick + round(start_seconds * self.tick_rate)
        upper = round_info.freeze_end_tick + round(end_seconds * self.tick_rate)
        if instant:
            nearest = np.argmin(np.abs(self.pose_ticks[candidate_ids] - lower))
            selected_ids = candidate_ids[[nearest]]
        else:
            within = candidate_ids[
                (self.pose_ticks[candidate_ids] >= lower) & (self.pose_ticks[candidate_ids] <= upper)
            ]
            selected_ids = within[::tick_step]
        if not len(selected_ids):
            raise ValueError("No player poses exist inside the selected time window.")
        positions = self.pose_positions[selected_ids].astype(np.float64, copy=True)
        ducks = np.clip(self.pose_ducks[selected_ids].astype(np.float64), 0.0, 1.0)
        positions[:, 2] += eye_height + (crouch_eye_height - eye_height) * ducks
        return [
            PlayerPose(
                int(self.pose_ticks[row_id]), positions[index],
                float(self.pose_yaws[row_id]), float(self.pose_pitches[row_id]),
            )
            for index, row_id in enumerate(selected_ids)
        ]

    def analyze_vision(
        self, poses: list[PlayerPose], config: AnalysisConfig, progress_callback: Any = None,
    ) -> VisibilityTimelineResult:
        return self.map_context.visibility_analyzer(config).analyze_timeline(
            poses, show_progress=False, progress_callback=progress_callback,
        )

    def analyze_flash(
        self, flash: FlashDetonation, config: FlashCoverageConfig, progress_callback: Any = None,
    ) -> FlashCoverageResult:
        return self.map_context.flash_analyzer(config).analyze(
            flash, show_progress=False, progress_callback=progress_callback,
        )

    def geometry_glb(self) -> bytes:
        if self._geometry_glb is None:
            mesh = self.mesh.copy()
            mesh.visual.face_colors = np.tile((160, 160, 160, 255), (len(mesh.faces), 1))
            scene = trimesh.Scene()
            scene.add_geometry(mesh, node_name="map_geometry", geom_name="map_geometry")
            self._geometry_glb = export_glb_bytes(scene)
        return self._geometry_glb

    def to_archive(
        self, *, analysis_metadata: dict[str, Any] | None = None, analysis_data: bytes | None = None,
    ) -> bytes:
        """Save normalized poses, map geometry, and an optional replay result."""
        manifest = {
            "schemaVersion": SESSION_SCHEMA_VERSION,
            "metadata": self.metadata(),
            "players": [asdict(player) for player in self.players],
            "roundPlayers": [
                {"round": number, "player": player, **assignment}
                for (number, player), assignment in self.round_players.items()
            ],
            "analysis": analysis_metadata,
        }
        poses = BytesIO()
        np.savez_compressed(
            poses, ticks=self.pose_ticks, rounds=self.pose_rounds, players=self.pose_players,
            positions=self.pose_positions, yaws=self.pose_yaws, pitches=self.pose_pitches, ducks=self.pose_ducks,
        )
        geometry = BytesIO()
        np.savez_compressed(
            geometry, vertices=np.asarray(self.mesh.vertices, dtype=np.float32),
            faces=np.asarray(self.mesh.faces, dtype=np.uint32),
        )
        output = BytesIO()
        with zipfile.ZipFile(output, "w", allowZip64=True) as archive:
            archive.writestr("manifest.json", json.dumps(manifest, separators=(",", ":")))
            archive.writestr("poses.npz", poses.getvalue(), compress_type=zipfile.ZIP_STORED)
            archive.writestr("geometry.npz", geometry.getvalue(), compress_type=zipfile.ZIP_STORED)
            if analysis_metadata is not None and analysis_data is not None:
                archive.writestr("analysis.bin", analysis_data, compress_type=zipfile.ZIP_DEFLATED)
        return output.getvalue()

    @classmethod
    def from_archive_bytes(cls, payload: bytes, source_name: str = "session.cs2session") -> "DemoSession":
        with zipfile.ZipFile(BytesIO(payload), "r") as archive:
            required = {"manifest.json", "poses.npz", "geometry.npz"}
            if not required.issubset(archive.namelist()):
                raise ValueError("Session archive is missing required data.")
            manifest = json.loads(archive.read("manifest.json"))
            if manifest.get("schemaVersion") != SESSION_SCHEMA_VERSION:
                raise ValueError("Unsupported session archive version.")
            metadata = manifest["metadata"]
            with np.load(BytesIO(archive.read("poses.npz")), allow_pickle=False) as poses:
                pose_values = {name: poses[name].copy() for name in poses.files}
            with np.load(BytesIO(archive.read("geometry.npz")), allow_pickle=False) as geometry:
                vertices, faces = geometry["vertices"].copy(), geometry["faces"].copy()
            saved_analysis_data = archive.read("analysis.bin") if "analysis.bin" in archive.namelist() else None
        players = [PlayerInfo(**value) for value in manifest["players"]]
        round_players = {
            (int(value["round"]), int(value["player"])): {
                "side": value.get("side"), "team": value.get("team"),
            }
            for value in manifest["roundPlayers"]
        }
        rounds = [
            RoundInfo(
                int(value["number"]), int(value["startTick"]),
                int(value["freezeEndTick"]), int(value["endTick"]),
            )
            for value in metadata["rounds"]
        ]
        session = cls(
            source_name=source_name, map_name=str(metadata["mapName"]), tick_rate=float(metadata["tickRate"]),
            rounds=rounds, players=players, round_players=round_players,
            pose_ticks=pose_values["ticks"], pose_rounds=pose_values["rounds"],
            pose_players=pose_values["players"], pose_positions=pose_values["positions"],
            pose_yaws=pose_values["yaws"], pose_pitches=pose_values["pitches"],
            pose_ducks=pose_values["ducks"],
            flashes=[FlashDetonation.from_json_dict(value) for value in metadata.get("flashes", [])],
            mesh=trimesh.Trimesh(vertices=vertices, faces=faces, process=False),
        )
        session.saved_analysis_metadata = manifest.get("analysis")
        session.saved_analysis_data = saved_analysis_data
        return session
