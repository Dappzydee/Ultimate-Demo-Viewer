"""Unit tests for replay interchange and self-contained sessions."""

from __future__ import annotations

import unittest
import time

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
from viewer import ApplicationState


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

    def test_application_job_reuses_the_loaded_session(self) -> None:
        session = self.make_session()
        session.map_context._raycasters[False] = AllVisibleRaycaster()
        state = ApplicationState()
        state.session = session
        try:
            job = state.start_vision({
                "playerId": "steam:7", "roundNumber": 1, "timeMode": "instant",
                "startSeconds": 1, "endSeconds": 1, "tickStep": 4,
                "maxDistance": 30, "samplesPerTriangle": 1, "forceCpu": True,
            })
            for _ in range(100):
                if job.status in {"complete", "error"}:
                    break
                time.sleep(0.01)
            self.assertEqual(job.status, "complete", job.error)
            result = state.get_result(job.result_id)
            self.assertEqual(result.analysis_type, "vision")
            decoded = decode_visibility_timeline(result.data)
            self.assertEqual(decoded.face_count, 2)
            self.assertEqual(result.metadata["playerName"], "Player")
            self.assertEqual(state.export_result_glb(job.result_id, "cumulative", None)[:4], b"glTF")
            saved = state.export_session(job.result_id)
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
        finally:
            state.close()


if __name__ == "__main__":
    unittest.main()
