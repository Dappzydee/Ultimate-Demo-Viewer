"""Synthetic tests for grenade lineup derivation and interchange."""

from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from cs2_visibility.grenade_lineups import (
    IN_ATTACK,
    IN_ATTACK2,
    IN_JUMP,
    GrenadeLineupConfig,
    GrenadeRelease,
    ThrowPose,
    derive_grenade_lineup,
    discover_lineup_properties,
    export_lineup_commands,
)


def pose(
    tick: int, x: float, *, z: float = 0.0, pitch: float = -10.0, yaw: float = 90.0,
    buttons: int | None = IN_ATTACK, walking: bool | None = False,
    grounded: bool | None = True, duck: float = 0.0,
) -> ThrowPose:
    return ThrowPose(
        tick=tick, position=(x, 2.0, z), pitch=pitch, yaw=yaw,
        button_mask=buttons, walking=walking, grounded=grounded, duck_amount=duck,
    )


def release(samples: list[ThrowPose], grenade_type: str = "flashbang") -> GrenadeRelease:
    return GrenadeRelease(grenade_type, "7", "Player", 3, samples[-1].tick, samples[-1])


class GrenadeLineupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = GrenadeLineupConfig(
            tick_rate=10, stationary_speed=2, movement_speed=8,
            min_stationary_seconds=0.3, sustained_movement_seconds=0.2,
            walking_speed_threshold=15, collapse_distance=1, path_tick_step=2,
        )

    def test_finds_last_stationary_reference_before_sustained_run(self) -> None:
        samples = [pose(tick, 0) for tick in range(4)]
        samples.extend(pose(tick, float((tick - 3) * 2)) for tick in range(4, 9))
        lineup = derive_grenade_lineup(release(samples), samples, self.config)
        self.assertTrue(lineup.has_fixed_reference)
        self.assertEqual(lineup.reference_point.tick, 3)
        self.assertEqual(lineup.throw_type.movement, "running")
        self.assertEqual(lineup.throw_type.click, "left")
        self.assertEqual(lineup.setpos_command, "setpos 0 2 0")
        self.assertEqual(lineup.setang_command, "setang -10 90 0")
        self.assertEqual(
            lineup.movement_instruction,
            "Move right about 10.0 units from the reference while running; release with left-click at the release point.",
        )
        record = lineup.to_json_dict()
        self.assertEqual(record["T_release"], 8)
        self.assertEqual(record["reference_point"]["X"], 0)
        self.assertEqual(record["throw_type"]["movement"], "running")
        restored = type(lineup).from_json_dict(record)
        self.assertEqual(restored.movement_instruction, lineup.movement_instruction)
        self.assertEqual(restored.reference_point.position, lineup.reference_point.position)
        self.assertEqual(restored.throw_type, lineup.throw_type)

    def test_standing_throw_collapses_reference_to_release(self) -> None:
        samples = [pose(tick, 1) for tick in range(6)]
        lineup = derive_grenade_lineup(release(samples), samples, self.config)
        self.assertTrue(lineup.has_fixed_reference)
        self.assertEqual(lineup.reference_point.tick, samples[-1].tick)
        self.assertEqual(lineup.throw_type.movement, "standing")
        self.assertFalse(lineup.movement_path)

    def test_continuous_motion_uses_release_and_exports_path(self) -> None:
        samples = [pose(tick, float(tick * 2)) for tick in range(7)]
        lineup = derive_grenade_lineup(release(samples), samples, self.config)
        self.assertFalse(lineup.has_fixed_reference)
        self.assertIsNone(lineup.reference_point)
        self.assertEqual(lineup.setpos_command, "setpos 12 2 0")
        self.assertEqual(lineup.movement_path, ((0.0, 2.0, 0.0), (4.0, 2.0, 0.0), (8.0, 2.0, 0.0), (12.0, 2.0, 0.0)))
        self.assertIn("in-motion throw", " ".join(lineup.notes))
        self.assertIn("exported approach path", lineup.movement_instruction)

    def test_angle_drift_rejects_otherwise_valid_reference(self) -> None:
        samples = [pose(tick, 0, yaw=90) for tick in range(4)]
        samples.extend(pose(tick, float((tick - 3) * 2), yaw=120) for tick in range(4, 9))
        lineup = derive_grenade_lineup(release(samples), samples, self.config)
        self.assertFalse(lineup.has_fixed_reference)
        self.assertEqual(lineup.reference_angle_delta, (0.0, 30.0))
        self.assertIn("drift", " ".join(lineup.notes))

    def test_classifies_running_jumpthrow_and_both_click(self) -> None:
        samples = [pose(tick, float(tick * 2)) for tick in range(5)]
        samples[-1] = pose(
            4, 8, z=3, buttons=IN_ATTACK | IN_ATTACK2 | IN_JUMP,
            grounded=False,
        )
        lineup = derive_grenade_lineup(release(samples), samples, self.config)
        self.assertTrue(lineup.throw_type.jumpthrow)
        self.assertEqual(lineup.throw_type.movement, "running")
        self.assertEqual(lineup.throw_type.click, "both")
        self.assertEqual(lineup.throw_type.label, "running jumpthrow, left+right (medium)")

    def test_duck_note_is_written_to_command_export(self) -> None:
        samples = [pose(tick, 1, duck=1) for tick in range(4)]
        lineup = derive_grenade_lineup(release(samples), samples, self.config)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "commands.cfg"
            export_lineup_commands([lineup], output)
            text = output.read_text(encoding="utf-8")
        self.assertIn("// Crouch before reproducing this throw.", text)
        self.assertIn("// Stand at the reference point", text)
        self.assertIn("setpos 1 2 0", text)

    def test_discovers_current_demo_property_names(self) -> None:
        properties = discover_lineup_properties([
            "CCSPlayerPawn.CCSPlayer_MovementServices.m_nButtonDownMaskPrev",
            "CCSPlayerPawn.m_bIsWalking",
            "CCSPlayerPawn.m_hGroundEntity",
        ])
        self.assertEqual(properties["buttons"], "CCSPlayerPawn.CCSPlayer_MovementServices.m_nButtonDownMaskPrev")
        self.assertEqual(properties["walking"], "CCSPlayerPawn.m_bIsWalking")
        self.assertEqual(properties["ground"], "CCSPlayerPawn.m_hGroundEntity")
        self.assertIsNone(properties["velocity"])


if __name__ == "__main__":
    unittest.main()
