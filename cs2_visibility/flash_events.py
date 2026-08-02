"""Extraction and JSON interchange for exact flashbang detonation events."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import Any

import polars as pl
from awpy import Demo

from .progress import ProgressBar

LOGGER = logging.getLogger("cs2_visibility")


@dataclass(frozen=True)
class FlashDetonation:
    """A pop position reported by the demo's flashbang_detonate event."""

    index: int
    tick: int
    position: tuple[float, float, float]
    map_name: str | None = None
    round_number: int | None = None
    thrower: str | None = None
    thrower_steamid: str | None = None
    thrower_side: str | None = None
    thrower_team: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["position"] = {"x": self.position[0], "y": self.position[1], "z": self.position[2]}
        return result

    @classmethod
    def from_json_dict(cls, value: dict[str, Any]) -> "FlashDetonation":
        position = value["position"]
        if isinstance(position, dict):
            position = (position["x"], position["y"], position["z"])
        return cls(
            int(value["index"]), int(value["tick"]), tuple(map(float, position)),
            value.get("map_name"), value.get("round_number"), value.get("thrower"),
            value.get("thrower_steamid"), value.get("thrower_side"), value.get("thrower_team"),
        )


def _first_value(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return None


def extract_flash_detonations(demo_path: Path, show_progress: bool = True) -> list[FlashDetonation]:
    """Parse only detonation and required round events, then normalize flashes."""
    progress = ProgressBar(2, "Parsing flash detonation events", show_progress)
    progress.update(0)
    demo = Demo(str(demo_path), verbose=False)
    progress.update(1)
    # Calling Demo.parse() would additionally parse every player tick and all
    # grenade trajectories. This tool intentionally asks Awpy for the one
    # event stream needed for exact pop positions.
    event_data = demo.parse_events(["flashbang_detonate"])
    progress.update(2)
    event_frame = event_data.get("flashbang_detonate", pl.DataFrame())
    map_name = demo.header.get("map_name") or demo.header.get("map")
    LOGGER.debug("flashbang_detonate columns: %s", event_frame.columns)
    flashes: list[FlashDetonation] = []
    if event_frame.is_empty():
        progress.finish()
        return flashes
    for index, row in enumerate(event_frame.sort("tick").iter_rows(named=True)):
        x = _first_value(row, "X", "x")
        y = _first_value(row, "Y", "y")
        z = _first_value(row, "Z", "z")
        tick = _first_value(row, "tick")
        if x is None or y is None or z is None or tick is None:
            raise ValueError(f"flashbang_detonate row {index} is missing tick or X/Y/Z. Available columns: {event_frame.columns}")
        round_number = _first_value(row, "round_num", "round_number")
        flashes.append(FlashDetonation(
            index=index, tick=int(tick), position=(float(x), float(y), float(z)), map_name=str(map_name) if map_name else None,
            round_number=int(round_number) if round_number is not None else None,
            thrower=_first_value(row, "thrower", "attacker_name", "user_name", "name"),
            thrower_steamid=str(_first_value(row, "thrower_steamid", "attacker_steamid", "user_steamid")) if _first_value(row, "thrower_steamid", "attacker_steamid", "user_steamid") is not None else None,
            thrower_side=_first_value(row, "thrower_side", "attacker_side", "user_side", "side"),
            thrower_team=_first_value(row, "thrower_team_clan_name", "attacker_team_clan_name", "user_team_clan_name", "team_clan_name"),
        ))
    progress.finish()
    LOGGER.debug("Normalized %d flash detonation events from %s", len(flashes), demo_path)
    return flashes


def export_flash_detonations(flashes: list[FlashDetonation], output_path: Path) -> None:
    """Write portable selected-event input for future analysis jobs."""
    output_path.write_text(json.dumps([flash.to_json_dict() for flash in flashes], indent=2), encoding="utf-8")


def load_flash_detonation_json(path: Path, index: int | None = None) -> FlashDetonation:
    """Load one event object or select one record from an exported JSON list."""
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data if isinstance(data, list) else [data]
    if index is None:
        if len(records) != 1:
            raise ValueError("JSON contains multiple flash events; provide --flash-index.")
        return FlashDetonation.from_json_dict(records[0])
    for record in records:
        if int(record["index"]) == index:
            return FlashDetonation.from_json_dict(record)
    raise ValueError(f"Flash index {index} was not found in {path}.")
