"""Compact binary interchange between the Python analyzers and WebGL UI."""

from __future__ import annotations

from dataclasses import dataclass
import struct

import numpy as np

from .models import VisibilityTimelineResult


VISION_MAGIC = b"CSV1"
FLASH_MAGIC = b"CSF1"
FORMAT_VERSION = 1
_HEADER = struct.Struct("<4sIIIII")


@dataclass(frozen=True)
class DecodedVisionTimeline:
    face_count: int
    ticks: np.ndarray
    instant_masks: np.ndarray
    cumulative_masks: np.ndarray

    @property
    def final_mask(self) -> np.ndarray:
        if not len(self.cumulative_masks):
            return np.zeros(self.face_count, dtype=bool)
        return np.unpackbits(self.cumulative_masks[-1], bitorder="little")[: self.face_count].astype(bool)


def encode_visibility_timeline(result: VisibilityTimelineResult) -> bytes:
    """Encode ticks followed by current and accumulated packed face masks."""
    frame_count = len(result.ticks)
    packed_width = (result.face_count + 7) // 8
    if result.instant_masks.shape != (frame_count, packed_width):
        raise ValueError("Instant timeline dimensions do not match its face and frame counts.")
    if result.cumulative_masks.shape != (frame_count, packed_width):
        raise ValueError("Cumulative timeline dimensions do not match its face and frame counts.")
    header = _HEADER.pack(
        VISION_MAGIC, FORMAT_VERSION, result.face_count, frame_count, packed_width, 0,
    )
    return b"".join(
        (
            header,
            np.asarray(result.ticks, dtype="<u4").tobytes(order="C"),
            np.asarray(result.instant_masks, dtype=np.uint8).tobytes(order="C"),
            np.asarray(result.cumulative_masks, dtype=np.uint8).tobytes(order="C"),
        )
    )


def decode_visibility_timeline(payload: bytes) -> DecodedVisionTimeline:
    """Decode and validate a visibility timeline payload."""
    if len(payload) < _HEADER.size:
        raise ValueError("Visibility timeline is truncated.")
    magic, version, face_count, frame_count, packed_width, _ = _HEADER.unpack_from(payload)
    if magic != VISION_MAGIC or version != FORMAT_VERSION:
        raise ValueError("Unsupported visibility timeline format.")
    if packed_width != (face_count + 7) // 8:
        raise ValueError("Visibility timeline has an invalid packed-mask width.")
    ticks_size = frame_count * 4
    masks_size = frame_count * packed_width
    expected = _HEADER.size + ticks_size + masks_size * 2
    if len(payload) != expected:
        raise ValueError("Visibility timeline length does not match its header.")
    offset = _HEADER.size
    ticks = np.frombuffer(payload, dtype="<u4", count=frame_count, offset=offset).copy()
    offset += ticks_size
    instant = np.frombuffer(payload, dtype=np.uint8, count=masks_size, offset=offset).reshape(frame_count, packed_width).copy()
    offset += masks_size
    cumulative = np.frombuffer(payload, dtype=np.uint8, count=masks_size, offset=offset).reshape(frame_count, packed_width).copy()
    return DecodedVisionTimeline(face_count, ticks, instant, cumulative)


def encode_flash_intensities(intensities: np.ndarray) -> bytes:
    """Quantize normalized flash intensities to one byte per face."""
    values = np.rint(np.clip(intensities, 0.0, 1.0) * 255).astype(np.uint8)
    return _HEADER.pack(FLASH_MAGIC, FORMAT_VERSION, len(values), 1, len(values), 0) + values.tobytes()


def decode_flash_intensities(payload: bytes) -> np.ndarray:
    """Decode flash intensities back to normalized float values."""
    if len(payload) < _HEADER.size:
        raise ValueError("Flash result is truncated.")
    magic, version, face_count, frame_count, packed_width, _ = _HEADER.unpack_from(payload)
    if magic != FLASH_MAGIC or version != FORMAT_VERSION or frame_count != 1 or packed_width != face_count:
        raise ValueError("Unsupported flash result format.")
    if len(payload) != _HEADER.size + face_count:
        raise ValueError("Flash result length does not match its header.")
    return np.frombuffer(payload, dtype=np.uint8, offset=_HEADER.size).astype(np.float32) / 255.0
