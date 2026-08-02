"""Unit tests for replay interchange and self-contained sessions."""

from __future__ import annotations

from dataclasses import replace
from io import BytesIO
import json
import unittest
import time
import zipfile

import numpy as np
import trimesh

from cs2_visibility.analysis import VisibilityAnalyzer
from cs2_visibility.flash_events import FlashDetonation
from cs2_visibility.interchange import (
    decode_flash_intensities,
    decode_visibility_timeline,
    encode_visibility_timeline,
)
from cs2_visibility.models import AnalysisConfig, PlayerPose
from cs2_visibility.session import DemoSession, PlayerInfo, RoundInfo
from viewer import ApplicationState, StoredResult


class AllVisibleRaycaster:
    name = "test"

    def visible(self, origins, directions, face_ids, lengths):
        return np.ones(len(face_ids), dtype=bool)


def synthetic_mesh() -> trimesh.Trimesh:
    return trimesh.Trimesh(
        vertices=np.array(
            [
                [9, -1, -1], [9, 1, -1], [9, 0, 1],
                [-1, 9, -1], [1, 9, -1], [0, 9, 1],
            ],
            dtype=np.float64,
        ),
        faces=np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64),
        process=False,
    )


class VisibilityTimelineTests(unittest.TestCase):
    def test_current_and_accumulated_masks_remain_distinct(self) -> None:
        analyzer = VisibilityAnalyzer(
            synthetic_mesh(), AnalysisConfig(samples_per_triangle=1, prefer_gpu=False),
            sample_points=np.array([[10, 0, 0], [0, 10, 0]], dtype=np.float64),
            sample_face_ids=np.array([0, 1], dtype=np.int32), raycaster=AllVisibleRaycaster(),
        )
        result = analyzer.analyze_timeline(
            [
                PlayerPose(100, np.zeros(3), 0, 0),
                PlayerPose(104, np.zeros(3), 90, 0),
            ],
            show_progress=False,
        )
        current = np.unpackbits(result.instant_masks, axis=1, bitorder="little")[:, :2]
        cumulative = np.unpackbits(result.cumulative_masks, axis=1, bitorder="little")[:, :2]
        np.testing.assert_array_equal(current, [[1, 0], [0, 1]])
        np.testing.assert_array_equal(cumulative, [[1, 0], [1, 1]])
        decoded = decode_visibility_timeline(encode_visibility_timeline(result))
        np.testing.assert_array_equal(decoded.ticks, [100, 104])
        np.testing.assert_array_equal(decoded.final_mask, [True, True])
        np.testing.assert_array_equal(decoded.positions, np.zeros((2, 3)))
        np.testing.assert_array_equal(decoded.yaws, [0, 90])
        legacy = decode_visibility_timeline(encode_visibility_timeline(
            replace(result, positions=None, yaws=None, pitches=None),
        ))
        self.assertIsNone(legacy.positions)
        np.testing.assert_array_equal(legacy.final_mask, [True, True])


class SessionArchiveTests(unittest.TestCase):
    def make_session(self) -> DemoSession:
        return DemoSession(
            source_name="match.dem", map_name="de_test", tick_rate=64,
            rounds=[RoundInfo(1, 0, 64, 704)],
            players=[PlayerInfo("steam:7", "Player", "7")],
            round_players={(1, 0): {"side": "ct", "team": "Example"}},
            pose_ticks=np.array([64, 128], dtype=np.int64),
            pose_rounds=np.array([1, 1], dtype=np.int16),
            pose_players=np.array([0, 0], dtype=np.int16),
            pose_positions=np.array([[0, 0, 0], [1, 2, 3]], dtype=np.float32),
            pose_yaws=np.array([0, 90], dtype=np.float32),
            pose_pitches=np.array([0, 5], dtype=np.float32),
            pose_ducks=np.array([0, 1], dtype=np.float32),
            flashes=[FlashDetonation(0, 128, (1, 2, 3), "de_test", 1, "Player", "7", "ct", "Example")],
            mesh=synthetic_mesh(),
        )

    def test_archive_reopens_without_the_demo(self) -> None:
        session = self.make_session()
        payload = session.to_archive(
            analysis_metadata={"analysisType": "flash", "source": "demo"}, analysis_data=b"saved-result",
        )
        restored = DemoSession.from_archive_bytes(payload, "saved.cs2session")
        self.assertEqual(restored.map_name, "de_test")
        self.assertEqual(restored.players[0].name, "Player")
        self.assertEqual(restored.flashes[0].thrower_team, "Example")
        self.assertEqual(restored.saved_analysis_data, b"saved-result")
        poses = restored.select_poses(
            "steam:7", 1, 1, 1, instant=True, tick_step=4, eye_height=64, crouch_eye_height=46,
        )
        self.assertEqual(poses[0].tick, 128)
        self.assertAlmostEqual(poses[0].position[2], 49.0)
        hit = restored.map_context.pick_surface(np.zeros(3), np.array([1, 0, 0]))
        np.testing.assert_allclose(hit, [9, 0, 0])

    def test_schema_one_archive_remains_supported(self) -> None:
        current = self.make_session().to_archive(
            analysis_metadata={"analysisType": "flash", "source": "legacy"},
            analysis_data=b"legacy-result",
        )
        legacy = BytesIO()
        with zipfile.ZipFile(BytesIO(current), "r") as source, zipfile.ZipFile(legacy, "w") as target:
            manifest = json.loads(source.read("manifest.json"))
            manifest["schemaVersion"] = 1
            manifest["analysis"] = manifest.pop("analyses")[0]
            target.writestr("manifest.json", json.dumps(manifest))
            target.writestr("poses.npz", source.read("poses.npz"))
            target.writestr("geometry.npz", source.read("geometry.npz"))
            target.writestr("analysis.bin", source.read("analyses/0.bin"))
        reopened = DemoSession.from_archive_bytes(legacy.getvalue(), "legacy.cs2session")
        self.assertEqual(reopened.saved_analysis_data, b"legacy-result")
        self.assertEqual(reopened.saved_analysis_metadata["source"], "legacy")

    def test_application_job_reuses_the_loaded_session(self) -> None:
        session = self.make_session()
        session.map_context._raycasters[False] = AllVisibleRaycaster()
        state = ApplicationState()
        state.session = session
        try:
            vision_request = {
                "playerId": "steam:7", "roundNumber": 1, "timeMode": "instant",
                "startSeconds": 1, "endSeconds": 1, "tickStep": 4,
                "maxDistance": 30, "samplesPerTriangle": 1, "forceCpu": True,
            }
            job = state.start_vision(vision_request)
            for _ in range(100):
                if job.status in {"complete", "error"}:
                    break
                time.sleep(0.01)
            self.assertEqual(job.status, "complete", job.error)
            result = state.get_result(job.result_id)
            self.assertEqual(result.analysis_type, "vision")
            decoded = decode_visibility_timeline(result.data)
            self.assertEqual(decoded.face_count, 2)
            np.testing.assert_allclose(decoded.positions, [[1, 2, 3]])
            np.testing.assert_allclose(decoded.yaws, [90])
            self.assertEqual(result.metadata["playerName"], "Player")
            self.assertEqual(state.export_result_glb(job.result_id, "cumulative", None)[:4], b"glTF")
            cached_job = state.start_vision(vision_request)
            self.assertEqual(cached_job.status, "complete")
            self.assertEqual(cached_job.result_id, job.result_id)
            self.assertEqual(len(state.results), 1)

            saved = state.export_session([job.result_id])
            reopened = DemoSession.from_archive_bytes(saved)
            self.assertEqual(reopened.saved_analysis_data, result.data)

            flash_job = state.start_flash({
                "eventIndex": 0, "roundNumber": 1, "maxDistance": 30,
                "samplesPerTriangle": 1, "forceCpu": True,
            })
            for _ in range(100):
                if flash_job.status in {"complete", "error"}:
                    break
                time.sleep(0.01)
            self.assertEqual(flash_job.status, "complete", flash_job.error)
            flash_result = state.get_result(flash_job.result_id)
            self.assertEqual(flash_result.analysis_type, "flash")
            self.assertEqual(len(decode_flash_intensities(flash_result.data)), 2)
            self.assertEqual(state.export_result_glb(flash_job.result_id, "cumulative", None)[:4], b"glTF")

            saved_all = DemoSession.from_archive_bytes(state.export_session(None))
            self.assertEqual(len(saved_all.saved_analyses), 2)
            discarded = state.set_result_discarded(flash_job.result_id, True)
            self.assertTrue(discarded.public_metadata()["discarded"])
            saved_without_discarded = DemoSession.from_archive_bytes(state.export_session(None))
            self.assertEqual(len(saved_without_discarded.saved_analyses), 1)
            state.set_result_discarded(flash_job.result_id, False)

            history_ids = set(state.results)
            preview_job = state.start_flash({
                "position": [2, 3, 4], "maxDistance": 30,
                "samplesPerTriangle": 4, "forceCpu": True,
            }, preview=True)
            for _ in range(100):
                if preview_job.status in {"complete", "error"}:
                    break
                time.sleep(0.01)
            self.assertEqual(preview_job.status, "complete", preview_job.error)
            preview_result = state.get_result(preview_job.result_id)
            self.assertTrue(preview_result.metadata["preview"])
            self.assertEqual(preview_result.metadata["config"]["samples_per_triangle"], 1)
            self.assertEqual(set(state.results), history_ids)
            self.assertNotIn(preview_job.result_id, {
                item["id"] for item in state.public_state()["results"]
            })

            saved_history = state.export_session([job.result_id, flash_job.result_id])
            reopened_history = DemoSession.from_archive_bytes(saved_history)
            self.assertEqual(len(reopened_history.saved_analyses), 2)
        finally:
            state.close()

    def test_history_evicts_old_unpinned_results_but_keeps_pins(self) -> None:
        state = ApplicationState()
        try:
            pinned = StoredResult("pinned", "flash", {}, b"x", pinned=True)
            state._store_result(pinned)
            for index in range(21):
                state._store_result(StoredResult(str(index), "flash", {}, b"x"))
            self.assertIn("pinned", state.results)
            self.assertNotIn("0", state.results)
            self.assertEqual(sum(not result.pinned for result in state.results.values()), 20)
            renamed = state.rename_result("1", "Useful flash")
            self.assertEqual(renamed.public_metadata()["name"], "Useful flash")
            state.set_result_pinned("1", True)
            self.assertTrue(state.get_result("1").pinned)
            state.set_result_discarded("1", True)
            self.assertTrue(state.get_result("1").discarded)
            self.assertFalse(state.get_result("1").pinned)
            state.set_result_discarded("1", False)
            state.delete_result("1")
            self.assertNotIn("1", state.results)
        finally:
            state.close()


if __name__ == "__main__":
    unittest.main()
