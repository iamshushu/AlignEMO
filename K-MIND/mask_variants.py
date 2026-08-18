"""
Helpers for three visual input variants used in Table 11 of AlignEMO:
    - scene:           original full frame                       (no masking)
    - focused_facial:  background zeroed out, only face kept     (apply_bbox_mask_to_frames)
    - masked_facial:   face region zeroed out, background kept   (apply_inverse_bbox_mask_to_frames)

The first two already exist in build_cache_2_final.py.
This file adds the third variant and a unified selector.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np


def apply_bbox_mask_to_frames(
    frames: List[np.ndarray],
    frame_keys: List[str],
    bbox: Optional[Dict[str, List[Tuple[int, int, int, int]]]],
) -> List[np.ndarray]:
    """Focused Facial: zero out everything EXCEPT the face bbox."""
    if bbox is None:
        return [f.copy() for f in frames]

    out = []
    for frame, fk in zip(frames, frame_keys):
        masked = np.zeros_like(frame)
        has = False
        if fk in bbox:
            H, W = frame.shape[:2]
            for (x, y, w, h) in bbox[fk]:
                if w <= 0 or h <= 0:
                    continue
                x1 = max(0, int(x)); y1 = max(0, int(y))
                x2 = min(W, int(x + w)); y2 = min(H, int(y + h))
                if x2 > x1 and y2 > y1:
                    masked[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
                    has = True
        out.append(masked if has else frame)
    return out


def apply_inverse_bbox_mask_to_frames(
    frames: List[np.ndarray],
    frame_keys: List[str],
    bbox: Optional[Dict[str, List[Tuple[int, int, int, int]]]],
) -> List[np.ndarray]:
    """Masked Facial: zero out the face bbox, keep everything else."""
    if bbox is None:
        return [f.copy() for f in frames]

    out = []
    for frame, fk in zip(frames, frame_keys):
        masked = frame.copy()
        if fk in bbox:
            H, W = frame.shape[:2]
            for (x, y, w, h) in bbox[fk]:
                if w <= 0 or h <= 0:
                    continue
                x1 = max(0, int(x)); y1 = max(0, int(y))
                x2 = min(W, int(x + w)); y2 = min(H, int(y + h))
                if x2 > x1 and y2 > y1:
                    masked[y1:y2, x1:x2] = 0
        out.append(masked)
    return out


# convenience dispatcher
VARIANTS = {
    "scene":           lambda frames, keys, bbox: [f.copy() for f in frames],
    "focused_facial":  apply_bbox_mask_to_frames,
    "masked_facial":   apply_inverse_bbox_mask_to_frames,
}
