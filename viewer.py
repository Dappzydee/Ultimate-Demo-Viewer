"""Launch the integrated local CS2 demo analyzer and GLB viewer."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, replace
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import time
from typing import Any, Callable
from urllib.parse import parse_qs, unquote, urlsplit
import uuid
import webbrowser

import numpy as np

from cs2_visibility.analysis import export_colored_mesh_bytes
from cs2_visibility.flash_coverage import FlashCoverageConfig, export_flash_coverage_bytes
from cs2_visibility.flash_events import FlashDetonation
from cs2_visibility.interchange import (
    decode_flash_intensities,
    decode_multi_visibility_timeline,
    decode_visibility_timeline,
    encode_flash_intensities,
    encode_multi_visibility_timeline,
)
from cs2_visibility.models import AnalysisConfig
from cs2_visibility.session import DemoSession


ROOT = Path(__file__).resolve().parent
VIEWER_ROOT = ROOT / "viewer"
MAX_UNPINNED_RESULTS = 20
MAX_RESULT_BYTES = 512 * 1024 * 1024


@dataclass
class StoredResult:
    id: str
    analysis_type: str
    metadata: dict[str, Any]
    data: bytes
    cache_key: str | None = None
    pinned: bool = False
    discarded: bool = False
    created_at: float = field(default_factory=time.time)

    def public_metadata(self) -> dict[str, Any]:
        return {
            **self.metadata,
            "id": self.id,
            "analysisType": self.analysis_type,
            "sizeBytes": len(self.data),
            "createdAt": self.created_at,
            "pinned": self.pinned,
            "discarded": self.discarded,
        }


@dataclass
class AnalysisJob:
    id: str
    kind: str
    status: str = "queued"
    progress: float = 0.0
    message: str = "Queued"
    result_id: str | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "resultId": self.result_id,
            "error": self.error,
        }


class ApplicationState:
    """Thread-safe application state owned by one local viewer server."""

    def __init__(self) -> None:
        self.session: DemoSession | None = None
        self.results: dict[str, StoredResult] = {}
        self.preview_results: dict[str, StoredResult] = {}
        self.jobs: dict[str, AnalysisJob] = {}
        self.lock = threading.RLock()
        self.working_directory = tempfile.TemporaryDirectory(prefix="cs2-viewer-")

    def close(self) -> None:
        self.working_directory.cleanup()

    def public_state(self) -> dict[str, Any]:
        with self.lock:
            total_bytes = sum(len(result.data) for result in self.results.values())
            pinned_bytes = sum(len(result.data) for result in self.results.values() if result.pinned)
            warning = None
            if total_bytes > MAX_RESULT_BYTES:
                warning = "Pinned or oversized results exceed the 512 MB history budget."
            return {
                "session": self.session.metadata() if self.session else None,
                "results": [result.public_metadata() for result in self.results.values()],
                "resultHistory": {
                    "totalBytes": total_bytes,
                    "pinnedBytes": pinned_bytes,
                    "maxBytes": MAX_RESULT_BYTES,
                    "maxUnpinnedResults": MAX_UNPINNED_RESULTS,
                    "warning": warning,
                },
            }

    @staticmethod
    def _cache_key(analysis_type: str, values: dict[str, Any]) -> str:
        return f"{analysis_type}:{json.dumps(values, sort_keys=True, separators=(',', ':'))}"

    def _cached_job(self, kind: str, cache_key: str) -> AnalysisJob | None:
        with self.lock:
            cached = next((
                result for result in self.results.values()
                if result.cache_key == cache_key and not result.discarded
            ), None)
            if cached is None:
                return None
            job = AnalysisJob(
                uuid.uuid4().hex, kind, status="complete", progress=1.0,
                message="Reused cached result", result_id=cached.id,
            )
            self.jobs[job.id] = job
            return job

    def _store_result(self, result: StoredResult) -> None:
        with self.lock:
            self.results[result.id] = result
            self._evict_results(result.id)

    def _store_preview_result(self, result: StoredResult) -> None:
        """Keep only the newest transient result used by the live viewport preview."""
        with self.lock:
            self.preview_results.clear()
            self.preview_results[result.id] = result

    def _evict_results(self, protected_id: str | None = None) -> None:
        def remove_oldest_unpinned() -> bool:
            candidate = next((
                result_id for result_id, result in self.results.items()
                if not result.pinned and result_id != protected_id
            ), None)
            if candidate is None:
                return False
            del self.results[candidate]
            return True

        while sum(not result.pinned for result in self.results.values()) > MAX_UNPINNED_RESULTS:
            if not remove_oldest_unpinned():
                break
        while sum(len(result.data) for result in self.results.values()) > MAX_RESULT_BYTES:
            if not remove_oldest_unpinned():
                break

    def set_result_pinned(self, result_id: str, pinned: bool) -> StoredResult:
        with self.lock:
            result = self.get_result(result_id)
            result.pinned = bool(pinned)
            if not result.pinned:
                self._evict_results()
            return result

    def set_result_discarded(self, result_id: str, discarded: bool) -> StoredResult:
        with self.lock:
            result = self.results.get(result_id)
            if result is None:
                raise ValueError("Analysis result was not found.")
            result.discarded = bool(discarded)
            if result.discarded:
                result.pinned = False
                self._evict_results()
            return result

    def delete_result(self, result_id: str) -> None:
        with self.lock:
            self.get_result(result_id)
            del self.results[result_id]

    def rename_result(self, result_id: str, name: str) -> StoredResult:
        clean_name = str(name).strip()
        if not clean_name:
            raise ValueError("Result name cannot be empty.")
        if len(clean_name) > 80:
            raise ValueError("Result name cannot exceed 80 characters.")
        with self.lock:
            result = self.get_result(result_id)
            result.metadata["name"] = clean_name
            return result

    def _has_active_job(self) -> bool:
        return any(job.status in {"queued", "running"} for job in self.jobs.values())

    def _start_job(self, kind: str, action: Callable[[AnalysisJob], None]) -> AnalysisJob:
        with self.lock:
            if self._has_active_job():
                raise RuntimeError("Another demo load or analysis is already running.")
            job = AnalysisJob(uuid.uuid4().hex, kind)
            self.jobs[job.id] = job

        def run() -> None:
            job.status = "running"
            try:
                action(job)
                job.progress = 1.0
                job.status = "complete"
                job.message = "Complete"
            except Exception as error:  # The job endpoint returns a concise UI-safe failure.
                job.status = "error"
                job.error = str(error)
                job.message = "Failed"

        threading.Thread(target=run, name=f"cs2-{kind}-{job.id[:8]}", daemon=True).start()
        return job

    @staticmethod
    def _job_progress(job: AnalysisJob, message: str) -> Callable[[int, int], None]:
        def update(completed: int, total: int) -> None:
            job.progress = completed / total if total else 1.0
            job.message = message
        return update

    def start_demo_load(self, path: Path) -> AnalysisJob:
        def load(job: AnalysisJob) -> None:
            job.message = "Parsing demo and loading map geometry"
            session = DemoSession.from_demo(path)
            self._install_session(session)
        return self._start_job("load-demo", load)

    def start_session_load(self, payload: bytes, filename: str) -> AnalysisJob:
        def load(job: AnalysisJob) -> None:
            job.message = "Opening saved session"
            self._install_session(DemoSession.from_archive_bytes(payload, filename))
        return self._start_job("load-session", load)

    def _install_session(self, session: DemoSession) -> None:
        with self.lock:
            self.session = session
            self.results.clear()
            self.preview_results.clear()
            for saved_metadata, saved_data in session.saved_analyses:
                metadata = dict(saved_metadata)
                analysis_type = str(metadata.pop("analysisType"))
                metadata.pop("id", None)
                pinned = bool(metadata.pop("pinned", False))
                discarded = bool(metadata.pop("discarded", False))
                created_at = float(metadata.pop("createdAt", time.time()))
                metadata.pop("sizeBytes", None)
                result_id = uuid.uuid4().hex
                self.results[result_id] = StoredResult(
                    result_id, analysis_type, metadata, saved_data,
                    pinned=pinned, discarded=discarded, created_at=created_at,
                )
            self._evict_results()

    def _require_session(self) -> DemoSession:
        if self.session is None:
            raise ValueError("Load a demo or saved session first.")
        return self.session

    def start_vision(self, request: dict[str, Any]) -> AnalysisJob:
        session = self._require_session()
        time_mode = str(request.get("timeMode", "interval"))
        instant = time_mode == "instant"
        start_seconds = float(request.get("startSeconds", 0.0))
        end_seconds = start_seconds if instant else float(request.get("endSeconds", start_seconds))
        tick_step = int(request.get("tickStep", 4))
        eye_height = float(request.get("eyeHeight", 64.0))
        crouch_eye_height = float(request.get("crouchEyeHeight", 46.0))
        requested_players = request.get("playerIds") or [request.get("playerId")]
        player_ids = [str(value) for value in requested_players if value]
        if not player_ids:
            raise ValueError("Select at least one player.")
        if len(set(player_ids)) != len(player_ids):
            raise ValueError("Each selected player may only appear once.")
        vision_enabled = bool(request.get("visionEnabled", True))
        gap_enabled = bool(request.get("gapEnabled", False))
        if not vision_enabled and not gap_enabled:
            raise ValueError("Enable vision coloring, gap coloring, or both.")
        round_number = int(request["roundNumber"])
        config = AnalysisConfig(
            horizontal_fov_degrees=float(request.get("fov", 90.0)),
            max_distance=float(request.get("maxDistance", 4000.0)),
            samples_per_triangle=int(request.get("samplesPerTriangle", 4)),
            min_visible_samples=int(request.get("minVisibleSamples", 1)),
            ray_batch_size=int(request.get("rayBatchSize", 250_000)),
            prefer_gpu=not bool(request.get("forceCpu", False)),
        )
        cache_key = self._cache_key("vision", {
            "playerIds": player_ids, "roundNumber": round_number, "timeMode": time_mode,
            "startSeconds": start_seconds, "endSeconds": end_seconds, "tickStep": tick_step,
            "eyeHeight": eye_height, "crouchEyeHeight": crouch_eye_height,
            "visionEnabled": vision_enabled, "gapEnabled": gap_enabled,
            "config": asdict(config),
        })
        cached = self._cached_job("vision", cache_key)
        if cached:
            return cached

        def analyze(job: AnalysisJob) -> None:
            pose_sets = [session.select_poses(
                player_id, round_number, start_seconds, end_seconds, instant=instant,
                tick_step=tick_step, eye_height=eye_height, crouch_eye_height=crouch_eye_height,
            ) for player_id in player_ids]
            shared_ticks = [pose.tick for pose in pose_sets[0]]
            if any([pose.tick for pose in poses] != shared_ticks for poses in pose_sets[1:]):
                raise ValueError("Selected players do not have aligned demo ticks in this time window.")
            job.message = f"Analyzing {len(player_ids)} player(s) across {len(shared_ticks)} frame(s)"
            timelines = []
            gaps = []
            ray_count = 0
            total_stages = len(player_ids) * (2 if gap_enabled else 1)
            stage = 0

            def stage_progress(completed: int, total: int) -> None:
                job.progress = (stage + (completed / total if total else 1.0)) / total_stages

            for index, poses in enumerate(pose_sets):
                job.message = f"Player {index + 1}/{len(player_ids)}: raycasting vision"
                timeline = session.analyze_vision(poses, config, stage_progress)
                stage += 1
                timelines.append(timeline)
                ray_count += timeline.tested_rays
                if gap_enabled:
                    job.message = f"Player {index + 1}/{len(player_ids)}: raycasting exposure"
                    exposure = session.analyze_vision(
                        poses, replace(config, horizontal_fov_degrees=360.0), stage_progress,
                    )
                    stage += 1
                    ray_count += exposure.tested_rays
                    gaps.append(np.bitwise_and(exposure.instant_masks, np.bitwise_not(timeline.instant_masks)))
                else:
                    gaps.append(np.zeros_like(timeline.instant_masks))
            players = [next(item for item in session.players if item.id == player_id) for player_id in player_ids]
            metadata = {
                "timeMode": time_mode,
                "roundNumber": round_number,
                "playerIds": [player.id for player in players],
                "playerNames": [player.name for player in players],
                "playerId": players[0].id,
                "playerName": players[0].name,
                "startSeconds": start_seconds,
                "endSeconds": end_seconds,
                "tickRate": session.tick_rate,
                "frameCount": timelines[0].processed_poses,
                "faceCount": timelines[0].face_count,
                "testedRays": ray_count,
                "backend": timelines[0].backend,
                "visionEnabled": vision_enabled,
                "gapEnabled": gap_enabled,
                "config": asdict(config),
            }
            result_id = uuid.uuid4().hex
            self._store_result(StoredResult(
                result_id, "vision", metadata,
                encode_multi_visibility_timeline(timelines, gaps), cache_key=cache_key,
            ))
            job.result_id = result_id

        return self._start_job("vision", analyze)

    def start_flash(self, request: dict[str, Any], *, preview: bool = False) -> AnalysisJob:
        session = self._require_session()
        event_index = int(request["eventIndex"]) if request.get("eventIndex") is not None else None
        manual_position = None
        if event_index is None:
            position = request.get("position")
            if not isinstance(position, list) or len(position) != 3:
                raise ValueError("A manual flash requires an X/Y/Z position.")
            manual_position = tuple(float(value) for value in position)
        config = FlashCoverageConfig(
            max_distance=float(request.get("maxDistance", 1500.0)),
            samples_per_triangle=1 if preview else int(request.get("samplesPerTriangle", 4)),
            falloff_power=float(request.get("falloffPower", 1.0)),
            ray_batch_size=int(request.get("rayBatchSize", 250_000)),
            prefer_gpu=not bool(request.get("forceCpu", False)),
        )
        cache_key = self._cache_key("flash", {
            "eventIndex": event_index, "position": manual_position,
            "roundNumber": request.get("roundNumber"), "tick": int(request.get("tick", 0)),
            "config": asdict(config),
        })
        if not preview:
            cached = self._cached_job("flash", cache_key)
            if cached:
                return cached

        def analyze(job: AnalysisJob) -> None:
            if event_index is not None:
                flash = next((item for item in session.flashes if item.index == event_index), None)
                if flash is None:
                    raise ValueError(f"Flash event {event_index} does not exist in this session.")
                source = "demo"
            else:
                flash = FlashDetonation(
                    index=-1, tick=int(request.get("tick", 0)),
                    position=manual_position,
                    map_name=session.map_name, round_number=request.get("roundNumber"), thrower="Manual placement",
                )
                source = "manual"
            job.message = "Simulating flash coverage"
            coverage = session.analyze_flash(
                flash, config, self._job_progress(job, "Raycasting flash coverage"),
            )
            metadata = {
                "source": source,
                "preview": preview,
                "faceCount": len(coverage.intensities),
                "testedRays": coverage.tested_rays,
                "backend": coverage.backend,
                "flash": flash.to_json_dict(),
                "config": asdict(config),
            }
            result_id = uuid.uuid4().hex
            result = StoredResult(
                result_id, "flash", metadata, encode_flash_intensities(coverage.intensities),
                cache_key=None if preview else cache_key,
            )
            if preview:
                self._store_preview_result(result)
            else:
                self._store_result(result)
            job.result_id = result_id

        return self._start_job("flash-preview" if preview else "flash", analyze)

    def get_job(self, job_id: str) -> AnalysisJob:
        try:
            return self.jobs[job_id]
        except KeyError as error:
            raise ValueError("Analysis job was not found.") from error

    def get_result(self, result_id: str) -> StoredResult:
        with self.lock:
            result = self.results.get(result_id) or self.preview_results.get(result_id)
            if result is None:
                raise ValueError("Analysis result was not found.")
            return result

    def export_result_glb(self, result_id: str, mode: str, frame: int | None) -> bytes:
        session = self._require_session()
        result = self.get_result(result_id)
        if result.analysis_type == "vision":
            if result.data[:4] == b"CSV2":
                timeline = decode_multi_visibility_timeline(result.data)
                if not len(timeline.ticks):
                    raise ValueError("The vision result has no replay frames.")
                frame_index = len(timeline.ticks) - 1 if frame is None else max(0, min(frame, len(timeline.ticks) - 1))
                vision_source = timeline.instant_masks if mode == "instant" else timeline.cumulative_masks
                vision = np.any(np.unpackbits(vision_source[:, frame_index], axis=1, bitorder="little")[:, :timeline.face_count], axis=0)
                gap = np.any(np.unpackbits(timeline.gap_masks[:, frame_index], axis=1, bitorder="little")[:, :timeline.face_count], axis=0)
                colors = np.full((timeline.face_count, 4), (160, 160, 160, 255), dtype=np.uint8)
                if result.metadata.get("visionEnabled", True): colors[vision] = (35, 235, 105, 220)
                if result.metadata.get("gapEnabled", False): colors[gap] = (255, 70, 70, 205)
                colors[vision & gap] = (255, 220, 80, 255)
                mesh = session.mesh.copy(); mesh.visual.face_colors = colors
                from cs2_visibility.geometry import export_glb_bytes
                return export_glb_bytes(mesh)
            timeline = decode_visibility_timeline(result.data)
            if not len(timeline.ticks):
                raise ValueError("The vision result has no replay frames.")
            frame_index = len(timeline.ticks) - 1 if frame is None else max(0, min(frame, len(timeline.ticks) - 1))
            packed = timeline.instant_masks[frame_index] if mode == "instant" else timeline.cumulative_masks[frame_index]
            mask = np.unpackbits(packed, bitorder="little")[: timeline.face_count].astype(bool)
            return export_colored_mesh_bytes(session.mesh, mask)
        intensities = decode_flash_intensities(result.data)
        flash = FlashDetonation.from_json_dict(result.metadata["flash"])
        return export_flash_coverage_bytes(session.mesh, intensities, flash)

    def export_session(self, result_ids: list[str] | None) -> bytes:
        session = self._require_session()
        if result_ids is None:
            result_ids = [result.id for result in self.results.values() if not result.discarded]
        analyses = []
        for result_id in dict.fromkeys(result_ids):
            result = self.results.get(result_id)
            if result is None or result.discarded:
                continue
            analyses.append((result.public_metadata(), result.data))
        return session.to_archive(analyses=analyses)

    def pick(self, origin: list[float], direction: list[float]) -> list[float] | None:
        hit = self._require_session().map_context.pick_surface(
            np.asarray(origin, dtype=np.float64), np.asarray(direction, dtype=np.float64),
        )
        return hit.tolist() if hit is not None else None

    def player_preview(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._require_session().player_preview(
            str(request["playerId"]), int(request["roundNumber"]),
            tick_step=int(request.get("tickStep", 4)),
        )


class ViewerRequestHandler(BaseHTTPRequestHandler):
    """Serve the app plus its local analysis API."""

    server: "ViewerServer"

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        split = urlsplit(self.path)
        path = unquote(split.path)
        query = parse_qs(split.query)
        try:
            if path == "/viewer-config.json":
                payload: dict[str, Any] = {
                    "modelUrl": "/model.glb" if self.server.model_path else None,
                    "modelName": self.server.model_path.name if self.server.model_path else None,
                }
                if self.server.startup_job_id:
                    payload["startupJobId"] = self.server.startup_job_id
                self._send_json(payload)
                return
            if path == "/model.glb" and self.server.model_path:
                self._send_file(self.server.model_path, "model/gltf-binary")
                return
            if path == "/api/state":
                self._send_json(self.server.app_state.public_state())
                return
            if path == "/api/geometry.glb":
                session = self.server.app_state._require_session()
                self._send_bytes(session.geometry_glb(), "model/gltf-binary")
                return
            if path.startswith("/api/jobs/"):
                job_id = path.removeprefix("/api/jobs/")
                self._send_json(self.server.app_state.get_job(job_id).to_json())
                return
            if path.startswith("/api/results/") and path.endswith("/data.bin"):
                result_id = path.removeprefix("/api/results/").removesuffix("/data.bin").strip("/")
                result = self.server.app_state.get_result(result_id)
                self._send_bytes(result.data, "application/octet-stream")
                return
            if path.startswith("/api/results/") and path.endswith("/export.glb"):
                result_id = path.removeprefix("/api/results/").removesuffix("/export.glb").strip("/")
                frame = int(query["frame"][0]) if "frame" in query else None
                mode = query.get("mode", ["cumulative"])[0]
                result = self.server.app_state.get_result(result_id)
                payload = self.server.app_state.export_result_glb(result_id, mode, frame)
                self._send_bytes(
                    payload, "model/gltf-binary",
                    disposition=f'attachment; filename="{result.analysis_type}_result.glb"',
                )
                return
            if path == "/api/session/export.cs2session":
                payload = self.server.app_state.export_session(query.get("result"))
                self._send_bytes(
                    payload, "application/zip",
                    disposition='attachment; filename="analysis.cs2session"',
                )
                return

            relative = "index.html" if path == "/" else path.lstrip("/")
            candidate = (VIEWER_ROOT / relative).resolve()
            if VIEWER_ROOT not in candidate.parents and candidate != VIEWER_ROOT:
                self.send_error(403)
                return
            if not candidate.is_file():
                self.send_error(404)
                return
            content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
            if candidate.suffix in {".js", ".css", ".html"}:
                content_type += "; charset=utf-8"
            self._send_file(candidate, content_type)
        except (ValueError, FileNotFoundError) as error:
            self._send_json({"error": str(error)}, 404)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = unquote(urlsplit(self.path).path)
        try:
            if path == "/api/demo/load":
                filename = Path(unquote(self.headers.get("X-Filename", "demo.dem"))).name
                if not filename.lower().endswith(".dem"):
                    raise ValueError("Choose a Counter-Strike demo with the .dem extension.")
                destination = Path(self.server.app_state.working_directory.name) / filename
                self._receive_file(destination)
                job = self.server.app_state.start_demo_load(destination)
                self._send_json(job.to_json(), 202)
                return
            if path == "/api/session/load":
                filename = Path(unquote(self.headers.get("X-Filename", "session.cs2session"))).name
                job = self.server.app_state.start_session_load(self._read_body(), filename)
                self._send_json(job.to_json(), 202)
                return
            if path == "/api/analyze/vision":
                job = self.server.app_state.start_vision(self._read_json())
                self._send_json(job.to_json(), 202)
                return
            if path == "/api/analyze/flash":
                job = self.server.app_state.start_flash(self._read_json())
                self._send_json(job.to_json(), 202)
                return
            if path == "/api/preview/flash":
                job = self.server.app_state.start_flash(self._read_json(), preview=True)
                self._send_json(job.to_json(), 202)
                return
            if path == "/api/preview/player":
                self._send_json(self.server.app_state.player_preview(self._read_json()))
                return
            if path.startswith("/api/results/") and path.endswith("/pin"):
                result_id = path.removeprefix("/api/results/").removesuffix("/pin").strip("/")
                result = self.server.app_state.set_result_pinned(
                    result_id, bool(self._read_json().get("pinned", True)),
                )
                self._send_json(result.public_metadata())
                return
            if path.startswith("/api/results/") and path.endswith("/discard"):
                result_id = path.removeprefix("/api/results/").removesuffix("/discard").strip("/")
                result = self.server.app_state.set_result_discarded(
                    result_id, bool(self._read_json().get("discarded", True)),
                )
                self._send_json(result.public_metadata())
                return
            if path.startswith("/api/results/") and path.endswith("/delete"):
                result_id = path.removeprefix("/api/results/").removesuffix("/delete").strip("/")
                self._read_json()
                self.server.app_state.delete_result(result_id)
                self._send_json({"deleted": result_id})
                return
            if path.startswith("/api/results/") and path.endswith("/rename"):
                result_id = path.removeprefix("/api/results/").removesuffix("/rename").strip("/")
                result = self.server.app_state.rename_result(result_id, self._read_json().get("name", ""))
                self._send_json(result.public_metadata())
                return
            if path == "/api/pick":
                request = self._read_json()
                self._send_json({"position": self.server.app_state.pick(request["origin"], request["direction"])})
                return
            self._send_json({"error": "Unknown API endpoint."}, 404)
        except RuntimeError as error:
            self._send_json({"error": str(error)}, 409)
        except (KeyError, TypeError, ValueError, FileNotFoundError, json.JSONDecodeError) as error:
            self._send_json({"error": str(error)}, 400)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            raise ValueError("The request body is empty.")
        return self.rfile.read(length)

    def _read_json(self) -> dict[str, Any]:
        return json.loads(self._read_body())

    def _receive_file(self, path: Path) -> None:
        remaining = int(self.headers.get("Content-Length", "0"))
        if remaining <= 0:
            raise ValueError("The uploaded demo is empty.")
        with path.open("wb") as output:
            while remaining:
                chunk = self.rfile.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise ValueError("The demo upload ended unexpectedly.")
                output.write(chunk)
                remaining -= len(chunk)

    def _send_file(self, path: Path, content_type: str) -> None:
        try:
            size = path.stat().st_size
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(size))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            with path.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_json(self, value: Any, status: int = 200) -> None:
        self._send_bytes(json.dumps(value).encode("utf-8"), "application/json; charset=utf-8", status)

    def _send_bytes(
        self, data: bytes, content_type: str, status: int = 200, disposition: str | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        if disposition:
            self.send_header("Content-Disposition", disposition)
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, format: str, *args: object) -> None:
        return


class ViewerServer(ThreadingHTTPServer):
    model_path: Path | None

    def __init__(
        self, address: tuple[str, int], model_path: Path | None,
        startup_path: Path | None = None,
    ) -> None:
        super().__init__(address, ViewerRequestHandler)
        self.model_path = model_path
        self.app_state = ApplicationState()
        self.startup_job_id: str | None = None
        if startup_path:
            if startup_path.suffix.lower() == ".dem":
                self.startup_job_id = self.app_state.start_demo_load(startup_path).id
            else:
                self.startup_job_id = self.app_state.start_session_load(
                    startup_path.read_bytes(), startup_path.name,
                ).id

    def server_close(self) -> None:
        super().server_close()
        self.app_state.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Open the integrated CS2 demo analyzer and result viewer.")
    parser.add_argument("input", nargs="?", type=Path, help="Optional .dem, .cs2session, or .glb to open initially")
    parser.add_argument("--port", type=int, default=0, help="Local port (default: choose an available port)")
    parser.add_argument("--no-open", action="store_true", help="Do not open the default browser")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = args.input.resolve() if args.input else None
    if input_path and (not input_path.is_file() or input_path.suffix.lower() not in {".dem", ".cs2session", ".glb"}):
        raise SystemExit(f"Supported input file not found: {input_path}")
    if not 0 <= args.port <= 65535:
        raise SystemExit("--port must be between 0 and 65535")
    model_path = input_path if input_path and input_path.suffix.lower() == ".glb" else None
    startup_path = input_path if input_path and input_path.suffix.lower() in {".dem", ".cs2session"} else None
    server = ViewerServer(("127.0.0.1", args.port), model_path, startup_path)
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"CS2 analyzer running at {url}")
    print("Press Ctrl+C to stop.")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nAnalyzer stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
