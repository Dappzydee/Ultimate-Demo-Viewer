"""Compact binary interchange between the Python analyzers and WebGL UI."""

from __future__ import annotations

from dataclasses import dataclass
import struct

import numpy as np

from .models import VisibilityTimelineResult


VISION_MAGIC = b"CSV1"
FLASH_MAGIC = b"CSF1"
VISION_FORMAT_VERSION = 3
MULTI_VISION_MAGIC = b"CSV2"
MULTI_VISION_FORMAT_VERSION = 1
FLASH_FORMAT_VERSION = 1
VISION_POSE_WIDTH = 6
VISION_POSE_WIDTHS = {2: 5, 3: 6}
_HEADER = struct.Struct("<4sIIIII")


@dataclass(frozen=True)
class DecodedMultiVisionTimeline:
    face_count: int
    ticks: np.ndarray
    instant_masks: np.ndarray
    cumulative_masks: np.ndarray
    gap_masks: np.ndarray
    poses: np.ndarray


def encode_multi_visibility_timeline(
    timelines: list[VisibilityTimelineResult], gap_masks: list[np.ndarray],
) -> bytes:
    """Encode aligned per-player vision and frame-local exposure-gap masks."""
    if not timelines or len(timelines) != len(gap_masks):
        raise ValueError("A multi-player result requires matching timelines and gap masks.")
    first = timelines[0]
    frame_count, face_count = len(first.ticks), first.face_count
    width = (face_count + 7) // 8
    blocks = []
    for timeline, gaps in zip(timelines, gap_masks, strict=True):
        if timeline.face_count != face_count or not np.array_equal(timeline.ticks, first.ticks):
            raise ValueError("Multi-player vision timelines must have identical ticks and geometry.")
        if gaps.shape != (frame_count, width):
            raise ValueError("Gap timeline dimensions do not match the vision timeline.")
        if timeline.positions is None or timeline.yaws is None or timeline.pitches is None:
            raise ValueError("Multi-player vision timelines require player poses.")
        ducks = np.zeros(frame_count, dtype=np.float32) if timeline.ducks is None else timeline.ducks
        poses = np.column_stack((timeline.positions, timeline.yaws, timeline.pitches, ducks)).astype("<f4")
        blocks.extend((
            poses.tobytes(order="C"),
            np.asarray(timeline.instant_masks, dtype=np.uint8).tobytes(order="C"),
            np.asarray(timeline.cumulative_masks, dtype=np.uint8).tobytes(order="C"),
            np.asarray(gaps, dtype=np.uint8).tobytes(order="C"),
        ))
    return b"".join((
        _HEADER.pack(MULTI_VISION_MAGIC, MULTI_VISION_FORMAT_VERSION, face_count, frame_count, width, len(timelines)),
        np.asarray(first.ticks, dtype="<u4").tobytes(order="C"),
        *blocks,
    ))


def decode_multi_visibility_timeline(payload: bytes) -> DecodedMultiVisionTimeline:
    """Decode the CSV2 multi-player vision/gap interchange format."""
    if len(payload) < _HEADER.size:
        raise ValueError("Multi-player visibility timeline is truncated.")
    magic, version, face_count, frame_count, width, player_count = _HEADER.unpack_from(payload)
    if magic != MULTI_VISION_MAGIC or version != MULTI_VISION_FORMAT_VERSION or not player_count:
        raise ValueError("Unsupported multi-player visibility timeline format.")
    if width != (face_count + 7) // 8:
        raise ValueError("Multi-player visibility timeline has an invalid mask width.")
    ticks_size = frame_count * 4
    pose_size = frame_count * VISION_POSE_WIDTH * 4
    mask_size = frame_count * width
    expected = _HEADER.size + ticks_size + player_count * (pose_size + mask_size * 3)
    if len(payload) != expected:
        raise ValueError("Multi-player visibility timeline length does not match its header.")
    offset = _HEADER.size
    ticks = np.frombuffer(payload, dtype="<u4", count=frame_count, offset=offset).copy()
    offset += ticks_size
    poses, instant, cumulative, gaps = [], [], [], []
    for _ in range(player_count):
        poses.append(np.frombuffer(payload, dtype="<f4", count=frame_count * VISION_POSE_WIDTH, offset=offset).reshape(frame_count, VISION_POSE_WIDTH).copy())
        offset += pose_size
        player_masks = []
        for _mask_kind in range(3):
            player_masks.append(np.frombuffer(payload, dtype=np.uint8, count=mask_size, offset=offset).reshape(frame_count, width).copy())
            offset += mask_size
        instant.append(player_masks[0]); cumulative.append(player_masks[1]); gaps.append(player_masks[2])
    return DecodedMultiVisionTimeline(
        face_count, ticks, np.asarray(instant), np.asarray(cumulative), np.asarray(gaps), np.asarray(poses),
    )


@dataclass(frozen=True)
class DecodedVisionTimeline:
    face_count: int
    ticks: np.ndarray
    instant_masks: np.ndarray
    cumulative_masks: np.ndarray
    positions: np.ndarray | None = None
    yaws: np.ndarray | None = None
    pitches: np.ndarray | None = None
    ducks: np.ndarray | None = None

    @property
    def final_mask(self) -> np.ndarray:
        if not len(self.cumulative_masks):
            return np.zeros(self.face_count, dtype=bool)
        return np.unpackbits(self.cumulative_masks[-1], bitorder="little")[: self.face_count].astype(bool)


def encode_visibility_timeline(result: VisibilityTimelineResult) -> bytes:
    """Encode ticks, optional frame poses, and packed visibility masks."""
    frame_count = len(result.ticks)
    packed_width = (result.face_count + 7) // 8
    if result.instant_masks.shape != (frame_count, packed_width):
        raise ValueError("Instant timeline dimensions do not match its face and frame counts.")
    if result.cumulative_masks.shape != (frame_count, packed_width):
        raise ValueError("Cumulative timeline dimensions do not match its face and frame counts.")
    pose_values = (result.positions, result.yaws, result.pitches)
    has_poses = all(value is not None for value in pose_values)
    if any(value is not None for value in pose_values) and not has_poses:
        raise ValueError("Vision pose positions, yaws, and pitches must be provided together.")
    pose_records = b""
    version = 1
    pose_width = 0
    if has_poses:
        positions = np.asarray(result.positions, dtype=np.float32)
        yaws = np.asarray(result.yaws, dtype=np.float32)
        pitches = np.asarray(result.pitches, dtype=np.float32)
        ducks = np.zeros(frame_count, dtype=np.float32) if result.ducks is None else np.asarray(result.ducks, dtype=np.float32)
        if positions.shape != (frame_count, 3) or yaws.shape != (frame_count,) or pitches.shape != (frame_count,):
            raise ValueError("Vision pose arrays do not match the timeline frame count.")
        if ducks.shape != (frame_count,):
            raise ValueError("Vision duck amounts do not match the timeline frame count.")
        version = VISION_FORMAT_VERSION
        pose_width = VISION_POSE_WIDTH
        pose_records = np.column_stack((positions, yaws, pitches, ducks)).astype("<f4").tobytes(order="C")
    header = _HEADER.pack(VISION_MAGIC, version, result.face_count, frame_count, packed_width, pose_width)
    return b"".join(
        (
            header,
            np.asarray(result.ticks, dtype="<u4").tobytes(order="C"),
            pose_records,
            np.asarray(result.instant_masks, dtype=np.uint8).tobytes(order="C"),
            np.asarray(result.cumulative_masks, dtype=np.uint8).tobytes(order="C"),
        )
    )


def decode_visibility_timeline(payload: bytes) -> DecodedVisionTimeline:
    """Decode and validate a visibility timeline payload."""
    if len(payload) < _HEADER.size:
        raise ValueError("Visibility timeline is truncated.")
    magic, version, face_count, frame_count, packed_width, pose_width = _HEADER.unpack_from(payload)
    if magic != VISION_MAGIC or version not in (1, *VISION_POSE_WIDTHS):
        raise ValueError("Unsupported visibility timeline format.")
    if packed_width != (face_count + 7) // 8:
        raise ValueError("Visibility timeline has an invalid packed-mask width.")
    ticks_size = frame_count * 4
    if version == 1 and pose_width != 0:
        raise ValueError("Legacy visibility timelines cannot contain pose records.")
    if version in VISION_POSE_WIDTHS and pose_width != VISION_POSE_WIDTHS[version]:
        raise ValueError("Visibility timeline has an invalid pose-record width.")
    poses_size = frame_count * pose_width * 4
    masks_size = frame_count * packed_width
    expected = _HEADER.size + ticks_size + poses_size + masks_size * 2
    if len(payload) != expected:
        raise ValueError("Visibility timeline length does not match its header.")
    offset = _HEADER.size
    ticks = np.frombuffer(payload, dtype="<u4", count=frame_count, offset=offset).copy()
    offset += ticks_size
    positions = yaws = pitches = ducks = None
    if pose_width:
        poses = np.frombuffer(
            payload, dtype="<f4", count=frame_count * pose_width, offset=offset,
        ).reshape(frame_count, pose_width).copy()
        positions, yaws, pitches = poses[:, :3], poses[:, 3], poses[:, 4]
        ducks = poses[:, 5] if pose_width >= 6 else np.zeros(frame_count, dtype=np.float32)
        offset += poses_size
    instant = np.frombuffer(payload, dtype=np.uint8, count=masks_size, offset=offset).reshape(frame_count, packed_width).copy()
    offset += masks_size
    cumulative = np.frombuffer(payload, dtype=np.uint8, count=masks_size, offset=offset).reshape(frame_count, packed_width).copy()
    return DecodedVisionTimeline(face_count, ticks, instant, cumulative, positions, yaws, pitches, ducks)


def encode_flash_intensities(intensities: np.ndarray) -> bytes:
    """Quantize normalized flash intensities to one byte per face."""
    values = np.rint(np.clip(intensities, 0.0, 1.0) * 255).astype(np.uint8)
    return _HEADER.pack(FLASH_MAGIC, FLASH_FORMAT_VERSION, len(values), 1, len(values), 0) + values.tobytes()


def decode_flash_intensities(payload: bytes) -> np.ndarray:
    """Decode flash intensities back to normalized float values."""
    if len(payload) < _HEADER.size:
        raise ValueError("Flash result is truncated.")
    magic, version, face_count, frame_count, packed_width, _ = _HEADER.unpack_from(payload)
    if magic != FLASH_MAGIC or version != FLASH_FORMAT_VERSION or frame_count != 1 or packed_width != face_count:
        raise ValueError("Unsupported flash result format.")
    if len(payload) != _HEADER.size + face_count:
        raise ValueError("Flash result length does not match its header.")
    return np.frombuffer(payload, dtype=np.uint8, offset=_HEADER.size).astype(np.float32) / 255.0
