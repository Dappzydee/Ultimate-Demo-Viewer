"""Grenade-release extraction and reproducible lineup derivation.

Awpy/demoparser column names are intentionally contained in this module.  The
derivation functions operate on typed records so they can be tested without a
demo and remain stable when a CS2 update changes a network property name.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import logging
import math
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
from awpy import Demo
from awpy.parsers.rounds import create_round_df

from .progress import ProgressBar


LOGGER = logging.getLogger("cs2_visibility")

# Source/Source 2 input bits present in m_nButtonDownMaskPrev.
IN_ATTACK = 1 << 0
IN_JUMP = 1 << 1
IN_ATTACK2 = 1 << 11

GRENADE_WEAPONS = {
    "weapon_flashbang": "flashbang",
    "weapon_hegrenade": "hegrenade",
    "weapon_smokegrenade": "smokegrenade",
    "weapon_molotov": "molotov",
    "weapon_incgrenade": "incendiary",
    "weapon_decoy": "decoy",
    "weapon_tagrenade": "tagrenade",
    "weapon_snowball": "snowball",
}

BUTTON_PROPERTIES = (
    "CCSPlayerPawn.CCSPlayer_MovementServices.m_nButtonDownMaskPrev",
    "CCSPlayerPawn.CCSPlayer_MovementServices.m_nButtons",
    "m_nButtonDownMaskPrev",
    "m_nButtons",
)
WALK_PROPERTIES = ("CCSPlayerPawn.m_bIsWalking", "m_bIsWalking")
GROUND_PROPERTIES = ("CCSPlayerPawn.m_hGroundEntity", "m_hGroundEntity")
VELOCITY_PROPERTIES = (
    "CCSPlayerPawn.m_vecAbsVelocity",
    "CCSPlayerPawn.m_vecVelocity",
    "m_vecAbsVelocity",
    "m_vecVelocity",
)
ROUND_EVENTS = ["round_start", "round_freeze_end", "round_end", "round_officially_ended"]


@dataclass(frozen=True)
class GrenadeLineupConfig:
    """Thresholds for finding a stable pre-movement lineup reference."""

    tick_rate: float = 64.0
    lookback_seconds: float = 5.0
    stationary_speed: float = 8.0
    movement_speed: float = 20.0
    walking_speed_threshold: float = 150.0
    jump_vertical_speed: float = 20.0
    min_stationary_seconds: float = 0.125
    sustained_movement_seconds: float = 0.09375
    minimum_moving_fraction: float = 0.7
    angle_stability_degrees: float = 5.0
    collapse_distance: float = 32.0
    crouch_threshold: float = 0.5
    path_tick_step: int = 4

    def __post_init__(self) -> None:
        positive = (
            "tick_rate", "lookback_seconds", "stationary_speed", "movement_speed",
            "walking_speed_threshold", "jump_vertical_speed", "min_stationary_seconds",
            "sustained_movement_seconds", "angle_stability_degrees", "collapse_distance",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.movement_speed <= self.stationary_speed:
            raise ValueError("movement_speed must be greater than stationary_speed.")
        if not 0 < self.minimum_moving_fraction <= 1:
            raise ValueError("minimum_moving_fraction must be in (0, 1].")
        if not 0 <= self.crouch_threshold <= 1:
            raise ValueError("crouch_threshold must be in [0, 1].")
        if self.path_tick_step < 1:
            raise ValueError("path_tick_step must be at least 1.")


@dataclass(frozen=True)
class ThrowPose:
    tick: int
    position: tuple[float, float, float]
    pitch: float
    yaw: float
    velocity: tuple[float, float, float] | None = None
    button_mask: int | None = None
    walking: bool | None = None
    grounded: bool | None = None
    duck_amount: float = 0.0

    def point_dict(self) -> dict[str, float]:
        return {
            "X": self.position[0], "Y": self.position[1], "Z": self.position[2],
            "pitch": self.pitch, "yaw": self.yaw, "duck_amount": self.duck_amount,
        }

    @classmethod
    def from_point_dict(cls, value: dict[str, Any], tick: int) -> "ThrowPose":
        return cls(
            tick=tick,
            position=(float(value["X"]), float(value["Y"]), float(value["Z"])),
            pitch=float(value["pitch"]), yaw=float(value["yaw"]),
            duck_amount=float(value.get("duck_amount", 0.0)),
        )


@dataclass(frozen=True)
class GrenadeRelease:
    grenade_type: str
    thrower_steamid: str
    thrower: str | None
    round_number: int | None
    release_tick: int
    pose: ThrowPose


@dataclass(frozen=True)
class ThrowType:
    click: Literal["left", "right", "both", "unknown"]
    movement: Literal["standing", "walking", "running"]
    jumpthrow: bool

    @property
    def label(self) -> str:
        click = {
            "left": "left-click (overhand)",
            "right": "right-click (underhand)",
            "both": "left+right (medium)",
            "unknown": "unknown click",
        }[self.click]
        if self.jumpthrow:
            movement = "jumpthrow" if self.movement == "standing" else f"{self.movement} jumpthrow"
        else:
            movement = self.movement
        return f"{movement}, {click}"


@dataclass(frozen=True)
class GrenadeLineup:
    grenade_type: str
    thrower_steamid: str
    thrower: str | None
    round_number: int | None
    release_tick: int
    release: ThrowPose
    reference_point: ThrowPose | None
    has_fixed_reference: bool
    movement_path: tuple[tuple[float, float, float], ...]
    throw_type: ThrowType
    movement_instruction: str
    release_velocity: tuple[float, float, float]
    reference_angle_delta: tuple[float, float] | None
    setpos_command: str
    setang_command: str
    notes: tuple[str, ...]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "grenade_type": self.grenade_type,
            "thrower_steamid": self.thrower_steamid,
            "thrower": self.thrower,
            "round": self.round_number,
            "T_release": self.release_tick,
            "release": self.release.point_dict(),
            "reference_point": self.reference_point.point_dict() if self.reference_point else None,
            "reference_tick": self.reference_point.tick if self.reference_point else None,
            "has_fixed_reference": self.has_fixed_reference,
            "movement_path": [
                {"X": point[0], "Y": point[1], "Z": point[2]} for point in self.movement_path
            ],
            "throw_type": {**asdict(self.throw_type), "label": self.throw_type.label},
            "movement_instruction": self.movement_instruction,
            "release_velocity": {
                "X": self.release_velocity[0], "Y": self.release_velocity[1], "Z": self.release_velocity[2],
            },
            "release_speed": math.hypot(self.release_velocity[0], self.release_velocity[1]),
            "reference_angle_delta": (
                {"pitch": self.reference_angle_delta[0], "yaw": self.reference_angle_delta[1]}
                if self.reference_angle_delta else None
            ),
            "setpos_command": self.setpos_command,
            "setang_command": self.setang_command,
            "notes": list(self.notes),
        }

    @classmethod
    def from_json_dict(cls, value: dict[str, Any]) -> "GrenadeLineup":
        release_tick = int(value["T_release"])
        reference_value = value.get("reference_point")
        angle_value = value.get("reference_angle_delta")
        velocity_value = value.get("release_velocity") or {}
        throw_value = value["throw_type"]
        return cls(
            grenade_type=str(value["grenade_type"]),
            thrower_steamid=str(value["thrower_steamid"]),
            thrower=value.get("thrower"),
            round_number=int(value["round"]) if value.get("round") is not None else None,
            release_tick=release_tick,
            release=ThrowPose.from_point_dict(value["release"], release_tick),
            reference_point=(
                ThrowPose.from_point_dict(reference_value, int(value.get("reference_tick") or release_tick))
                if reference_value else None
            ),
            has_fixed_reference=bool(value.get("has_fixed_reference")),
            movement_path=tuple(
                (float(point["X"]), float(point["Y"]), float(point["Z"]))
                for point in value.get("movement_path", [])
            ),
            throw_type=ThrowType(
                str(throw_value["click"]), str(throw_value["movement"]), bool(throw_value["jumpthrow"]),
            ),
            movement_instruction=str(value.get("movement_instruction") or "Movement instructions unavailable."),
            release_velocity=(
                float(velocity_value.get("X", 0.0)), float(velocity_value.get("Y", 0.0)),
                float(velocity_value.get("Z", 0.0)),
            ),
            reference_angle_delta=(
                (float(angle_value["pitch"]), float(angle_value["yaw"])) if angle_value else None
            ),
            setpos_command=str(value["setpos_command"]),
            setang_command=str(value["setang_command"]),
            notes=tuple(str(note) for note in value.get("notes", [])),
        )


def _angle_delta(left: float, right: float) -> float:
    return abs((left - right + 180.0) % 360.0 - 180.0)


def _computed_velocities(samples: Sequence[ThrowPose], tick_rate: float) -> np.ndarray:
    result = np.zeros((len(samples), 3), dtype=np.float64)
    for index, sample in enumerate(samples):
        if sample.velocity is not None and np.all(np.isfinite(sample.velocity)):
            result[index] = sample.velocity
            continue
        previous = index - 1 if index else (0 if len(samples) == 1 else 1)
        tick_delta = abs(samples[index].tick - samples[previous].tick)
        if tick_delta:
            sign = 1.0 if index else -1.0
            displacement = np.asarray(samples[index].position) - np.asarray(samples[previous].position)
            result[index] = displacement * (tick_rate / tick_delta) * sign
    return result


def _click_type(samples: Sequence[ThrowPose]) -> Literal["left", "right", "both", "unknown"]:
    masks = [sample.button_mask for sample in samples[-3:] if sample.button_mask is not None]
    combined = 0
    for mask in masks:
        combined |= int(mask)
    primary = bool(combined & IN_ATTACK)
    secondary = bool(combined & IN_ATTACK2)
    if primary and secondary:
        return "both"
    if secondary:
        return "right"
    if primary:
        return "left"
    return "unknown"


def _fixed_reference_index(samples: Sequence[ThrowPose], horizontal_speeds: np.ndarray, config: GrenadeLineupConfig) -> int | None:
    if horizontal_speeds[-1] <= config.stationary_speed:
        return len(samples) - 1
    minimum_stationary = max(2, math.ceil(config.min_stationary_seconds * config.tick_rate))
    sustained_movement = max(2, math.ceil(config.sustained_movement_seconds * config.tick_rate))
    stationary = horizontal_speeds <= config.stationary_speed
    moving = horizontal_speeds >= config.movement_speed
    for end in range(len(samples) - 2, minimum_stationary - 2, -1):
        if not stationary[end]:
            continue
        start = end
        while start > 0 and stationary[start - 1]:
            start -= 1
        if end - start + 1 < minimum_stationary:
            continue
        after = moving[end + 1 :]
        if len(after) < sustained_movement or not np.all(after[:sustained_movement]):
            continue
        if float(np.mean(after)) < config.minimum_moving_fraction:
            continue
        return end
    return None


def _movement_path(samples: Sequence[ThrowPose], step: int) -> tuple[tuple[float, float, float], ...]:
    selected = list(samples[::step])
    if selected[-1].tick != samples[-1].tick:
        selected.append(samples[-1])
    return tuple(sample.position for sample in selected)


def _format_number(value: float) -> str:
    value = 0.0 if abs(value) < 0.0000005 else value
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _movement_direction(reference: ThrowPose, release: ThrowPose) -> tuple[str, float]:
    delta = np.asarray(release.position[:2]) - np.asarray(reference.position[:2])
    distance = float(np.linalg.norm(delta))
    if distance <= 1e-9:
        return "in place", 0.0
    world_yaw = math.degrees(math.atan2(float(delta[1]), float(delta[0])))
    relative = (world_yaw - reference.yaw + 180.0) % 360.0 - 180.0
    sectors = (
        "forward", "forward-left", "left", "back-left",
        "back", "back-right", "right", "forward-right",
    )
    sector = int(round(relative / 45.0)) % 8
    return sectors[sector], distance


def _movement_instruction(
    fixed: bool,
    reference: ThrowPose | None,
    release: ThrowPose,
    throw_type: ThrowType,
) -> str:
    click = {
        "left": "left-click",
        "right": "right-click",
        "both": "left+right",
        "unknown": "the recorded click type",
    }[throw_type.click]
    jump = "jump and release" if throw_type.jumpthrow else "release"
    if not fixed:
        return f"Follow the exported approach path while {throw_type.movement}; {jump} with {click} at the release point."
    if reference is None:
        return f"Use a short {throw_type.movement} approach through the release point; {jump} with {click}."
    direction, distance = _movement_direction(reference, release)
    if throw_type.movement == "standing":
        return f"Stand at the reference point; {jump} with {click}."
    if distance <= 1e-9:
        return f"Use a short {throw_type.movement} approach through the release point; {jump} with {click}."
    return (
        f"Move {direction} about {distance:.1f} units from the reference while {throw_type.movement}; "
        f"{jump} with {click} at the release point."
    )


def derive_grenade_lineup(
    release: GrenadeRelease,
    samples: Sequence[ThrowPose],
    config: GrenadeLineupConfig,
) -> GrenadeLineup:
    """Derive one stable lineup reference and independent throw-type axes."""
    ordered = sorted((sample for sample in samples if sample.tick <= release.release_tick), key=lambda item: item.tick)
    ordered = [sample for index, sample in enumerate(ordered) if index == 0 or sample.tick != ordered[index - 1].tick]
    if not ordered or ordered[-1].tick != release.release_tick:
        ordered.append(release.pose)
    else:
        ordered[-1] = release.pose
    if len(ordered) < 2:
        raise ValueError(f"Grenade release at tick {release.release_tick} has insufficient pose history.")

    velocities = _computed_velocities(ordered, config.tick_rate)
    horizontal = np.linalg.norm(velocities[:, :2], axis=1)
    release_velocity = tuple(map(float, velocities[-1]))
    reference_index = _fixed_reference_index(ordered, horizontal, config)
    notes: list[str] = []
    angle_delta: tuple[float, float] | None = None
    reference: ThrowPose | None = None
    fixed = reference_index is not None

    if reference_index is not None:
        candidate = ordered[reference_index]
        angle_delta = (
            abs(candidate.pitch - release.pose.pitch),
            _angle_delta(candidate.yaw, release.pose.yaw),
        )
        if max(angle_delta) > config.angle_stability_degrees:
            fixed = False
            notes.append("yaw/pitch drift detected; no reliable fixed-aim reference")
        else:
            distance = float(np.linalg.norm(np.asarray(candidate.position) - np.asarray(release.pose.position)))
            if distance < config.collapse_distance:
                reference = release.pose
                notes.append("reference collapses to release point (standing/short movement)")
            else:
                reference = candidate
    else:
        notes.append("no fixed reference - in-motion throw")

    if not fixed:
        reference = None
    command_pose = reference if reference is not None else release.pose
    movement_path = () if fixed else _movement_path(ordered, config.path_tick_step)

    recent = ordered[-3:]
    click = _click_type(recent)
    release_walking = next((item.walking for item in reversed(recent) if item.walking is not None), None)
    release_speed = float(horizontal[-1])
    if release_speed <= config.stationary_speed:
        movement: Literal["standing", "walking", "running"] = "standing"
    elif release_walking is True or (release_walking is None and release_speed <= config.walking_speed_threshold):
        movement = "walking"
    else:
        movement = "running"
    jump_button = any(item.button_mask is not None and int(item.button_mask) & IN_JUMP for item in recent)
    airborne_rise = release.pose.grounded is False and release_velocity[2] > config.jump_vertical_speed
    jumpthrow = bool(jump_button or airborne_rise)
    throw_type = ThrowType(click, movement, jumpthrow)

    if click == "unknown":
        notes.append("click type unavailable from the parsed button mask")
    if click == "both":
        notes.append("both attack buttons held (medium-strength throw)")
    if command_pose.duck_amount >= config.crouch_threshold:
        notes.append("duck state active; crouch must be reproduced manually")
    if jumpthrow:
        notes.append("jump timing/movement must be reproduced manually")

    x, y, z = command_pose.position
    setpos = f"setpos {_format_number(x)} {_format_number(y)} {_format_number(z)}"
    setang = f"setang {_format_number(command_pose.pitch)} {_format_number(command_pose.yaw)} 0"
    movement_instruction = _movement_instruction(fixed, reference, release.pose, throw_type)
    return GrenadeLineup(
        grenade_type=release.grenade_type,
        thrower_steamid=release.thrower_steamid,
        thrower=release.thrower,
        round_number=release.round_number,
        release_tick=release.release_tick,
        release=release.pose,
        reference_point=reference,
        has_fixed_reference=fixed,
        movement_path=movement_path,
        throw_type=throw_type,
        movement_instruction=movement_instruction,
        release_velocity=release_velocity,
        reference_angle_delta=angle_delta,
        setpos_command=setpos,
        setang_command=setang,
        notes=tuple(notes),
    )


def _discover_property(fields: set[str], candidates: Sequence[str]) -> str | None:
    for candidate in candidates:
        if candidate in fields:
            return candidate
    return None


def discover_lineup_properties(updated_fields: Sequence[str]) -> dict[str, str | None]:
    """Resolve patch-dependent movement property names from one demo schema."""
    fields = set(updated_fields)
    return {
        "buttons": _discover_property(fields, BUTTON_PROPERTIES),
        "walking": _discover_property(fields, WALK_PROPERTIES),
        "ground": _discover_property(fields, GROUND_PROPERTIES),
        "velocity": _discover_property(fields, VELOCITY_PROPERTIES),
    }


def _row_value(row: dict[str, Any], name: str | None, *, event: bool = False) -> Any:
    if not name:
        return None
    names = (f"user_{name}", name) if event else (name,)
    for candidate in names:
        value = row.get(candidate)
        if value is not None:
            return value
    return None


def _finite_float(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _optional_int(value: Any) -> int | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if math.isfinite(number) else None


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "1"):
            return True
        if normalized in ("false", "0"):
            return False
        return None
    number = _finite_float(value)
    return bool(number) if number is not None else None


def _steamid(value: Any) -> str | None:
    if value in (None, 0, "0"):
        return None
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


def _grounded(value: Any) -> bool | None:
    if value is None:
        return None
    try:
        handle = int(value)
    except (TypeError, ValueError):
        return None
    return handle not in (-1, 0xFFFFFF, 0xFFFFFFFF)


def _vector(value: Any) -> tuple[float, float, float] | None:
    if value is None:
        return None
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    return result if len(result) == 3 and all(math.isfinite(item) for item in result) else None


def _pose_from_row(row: dict[str, Any], props: dict[str, str | None], *, event: bool = False) -> ThrowPose | None:
    prefix = "user_" if event else ""
    core = [_finite_float(row.get(f"{prefix}{name}")) for name in ("X", "Y", "Z", "pitch", "yaw")]
    if any(value is None for value in core) or row.get("tick") is None:
        return None
    x, y, z, pitch, yaw = core
    duck = _finite_float(row.get(f"{prefix}duck_amount"), 0.0) or 0.0
    mask_value = _row_value(row, props["buttons"], event=event)
    walking_value = _row_value(row, props["walking"], event=event)
    return ThrowPose(
        tick=int(row["tick"]), position=(float(x), float(y), float(z)),
        pitch=float(pitch), yaw=float(yaw),
        velocity=_vector(_row_value(row, props["velocity"], event=event)),
        button_mask=_optional_int(mask_value),
        walking=_optional_bool(walking_value),
        grounded=_grounded(_row_value(row, props["ground"], event=event)),
        duck_amount=float(np.clip(duck, 0.0, 1.0)),
    )


def _round_for_tick(tick: int, rounds: Sequence[dict[str, Any]]) -> int | None:
    for row in rounds:
        start = int(row["start"])
        end = row.get("official_end") if row.get("official_end") is not None else row.get("end")
        if start <= tick <= int(end):
            return int(row["round_num"])
    return None


def extract_grenade_lineups(
    demo_path: Path,
    config: GrenadeLineupConfig,
    *,
    grenade_types: set[str] | None = None,
    show_progress: bool = True,
) -> list[GrenadeLineup]:
    """Extract all matching grenade releases and derive human-usable lineups."""
    progress = ProgressBar(4, "Extracting grenade lineups", show_progress)
    demo = Demo(str(demo_path), verbose=False)
    props = discover_lineup_properties(demo.parser.list_updated_fields())
    LOGGER.debug("Lineup properties discovered: %s", props)
    if props["buttons"] is None:
        LOGGER.warning("No button mask property found; click type and button-based jumpthrow detection are unavailable.")
    if props["velocity"] is None:
        LOGGER.debug("No direct player velocity property found; deriving velocity from position deltas.")

    player_props = ["X", "Y", "Z", "pitch", "yaw", "duck_amount"]
    player_props.extend(value for value in props.values() if value is not None)
    weapon_events = demo.parse_events(["weapon_fire"], player_props=list(dict.fromkeys(player_props)))["weapon_fire"]
    progress.update(1)
    round_events = demo.parse_events(ROUND_EVENTS)
    rounds = [row for row in create_round_df(round_events).iter_rows(named=True)]
    progress.update(2)

    releases: list[GrenadeRelease] = []
    for row in weapon_events.sort("tick").iter_rows(named=True):
        weapon = str(row.get("weapon") or "").lower()
        grenade_type = GRENADE_WEAPONS.get(weapon)
        if grenade_type is None or (grenade_types and grenade_type not in grenade_types):
            continue
        pose = _pose_from_row(row, props, event=True)
        steamid = _steamid(row.get("user_steamid"))
        if pose is None or steamid is None:
            LOGGER.warning("Skipping %s at tick %s because thrower pose/SteamID is missing.", weapon, row.get("tick"))
            continue
        releases.append(GrenadeRelease(
            grenade_type=grenade_type,
            thrower_steamid=steamid,
            thrower=str(row["user_name"]) if row.get("user_name") else None,
            round_number=_round_for_tick(pose.tick, rounds),
            release_tick=pose.tick,
            pose=pose,
        ))
    if not releases:
        progress.finish()
        return []

    lookback_ticks = max(2, round(config.lookback_seconds * config.tick_rate))
    wanted_ticks = sorted({
        tick
        for release in releases
        for tick in range(max(0, release.release_tick - lookback_ticks), release.release_tick + 1)
    })
    player_ids = sorted({int(release.thrower_steamid) for release in releases})
    tick_frame = demo.parser.parse_ticks(
        wanted_props=list(dict.fromkeys(player_props)), players=player_ids, ticks=wanted_ticks,
    )
    progress.update(3)
    histories: dict[str, list[ThrowPose]] = {}
    for row in tick_frame.to_dict(orient="records"):
        steamid = _steamid(row.get("steamid"))
        pose = _pose_from_row(row, props)
        if steamid is not None and pose is not None:
            histories.setdefault(steamid, []).append(pose)
    for values in histories.values():
        values.sort(key=lambda item: item.tick)

    result: list[GrenadeLineup] = []
    for release in releases:
        lower = release.release_tick - lookback_ticks
        history = [
            sample for sample in histories.get(release.thrower_steamid, [])
            if lower <= sample.tick <= release.release_tick
        ]
        # Event-attached player properties can describe the preceding snapshot.
        # Prefer the exact tick pose, while preserving an event button mask when
        # that patch does not expose the mask through parse_ticks.
        exact_pose = next((sample for sample in reversed(history) if sample.tick == release.release_tick), None)
        effective_release = release
        if exact_pose is not None:
            exact_pose = replace(
                exact_pose,
                button_mask=exact_pose.button_mask if exact_pose.button_mask is not None else release.pose.button_mask,
                walking=exact_pose.walking if exact_pose.walking is not None else release.pose.walking,
                grounded=exact_pose.grounded if exact_pose.grounded is not None else release.pose.grounded,
            )
            effective_release = replace(release, pose=exact_pose)
        try:
            result.append(derive_grenade_lineup(effective_release, history, config))
        except ValueError as error:
            LOGGER.warning("Skipping %s at tick %d: %s", release.grenade_type, release.release_tick, error)
    progress.finish()
    LOGGER.debug("Derived %d lineups from %d grenade releases.", len(result), len(releases))
    return result


def export_grenade_lineups(lineups: Sequence[GrenadeLineup], output_path: Path) -> None:
    output_path.write_text(
        json.dumps([lineup.to_json_dict() for lineup in lineups], indent=2), encoding="utf-8",
    )


def lineup_command_block(lineup: GrenadeLineup, index: int | None = None) -> str:
    heading = f"// Lineup {index}: " if index is not None else "// "
    heading += f"{lineup.grenade_type} | {lineup.throw_type.label} | round {lineup.round_number or '?'}"
    lines = [heading]
    lines.append(f"// {lineup.movement_instruction}")
    if not lineup.has_fixed_reference:
        lines.append("// In-motion throw: commands use the release point; reproduce the approach manually.")
    if any("duck state active" in note for note in lineup.notes):
        lines.append("// Crouch before reproducing this throw.")
    lines.extend((lineup.setpos_command, lineup.setang_command))
    return "\n".join(lines)


def export_lineup_commands(lineups: Sequence[GrenadeLineup], output_path: Path) -> None:
    blocks = [lineup_command_block(lineup, index + 1) for index, lineup in enumerate(lineups)]
    output_path.write_text("\n\n".join(blocks) + ("\n" if blocks else ""), encoding="utf-8")
