"""
Single source of truth for Leap hand-pose preprocessing.

Both LeapDatasetLoader (training) and LeapPoseInference (live) import from here,
so the features the model sees at train time and at run time cannot drift apart.

Feature layout (63 floats = 21 joints x 3):
    joint 0        palm center
    joints 1-4     thumb  (tip, dip, pip, mcp)
    joints 5-8     index  (tip, dip, pip, mcp)
    joints 9-12    middle (tip, dip, pip, mcp)
    joints 13-16   ring   (tip, dip, pip, mcp)
    joints 17-20   pinky  (tip, dip, pip, mcp)

Any change to this file that alters the output numbers MUST bump
PREPROCESS_VERSION, which invalidates dataset caches and makes the inference
script refuse a model trained with a different version.
"""
import numpy as np

PREPROCESS_VERSION = 2
N_JOINTS = 21
FEATURE_DIM = N_JOINTS * 3            # 63
MIDDLE_MCP_IDX = 12                   # joint used as the pose-independent scale reference

DEFAULT_SCALE = "middle_mcp"          # "middle_mcp" | "max_dist" (the v1 behaviour)
DEFAULT_TRAIN_HAND = "right"          # hand the dataset was recorded with


def preprocess_tag(scale: str = DEFAULT_SCALE, train_hand: str = DEFAULT_TRAIN_HAND) -> str:
    """String stored in the dataset cache and the model config to detect mismatches."""
    return f"v{PREPROCESS_VERSION}|{scale}|{train_hand}"


def fill_missing(joints: np.ndarray, center: np.ndarray) -> np.ndarray:
    """
    Replace NaN (missing) joints with the palm center so they become (0, 0, 0)
    after centering, instead of a bogus offset.
    joints: (..., 20, 3) with NaN for missing; center: (3,) or (N, 3).
    """
    joints = np.asarray(joints, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    if joints.ndim == 3 and center.ndim == 2:
        center = center[:, None, :]
    return np.where(np.isnan(joints), center, joints)


def normalize_batch(
    X: np.ndarray,
    mirror: np.ndarray | None = None,
    scale: str = DEFAULT_SCALE,
) -> np.ndarray:
    """
    X:      (N, 63) raw coordinates (palm center first).
    mirror: optional (N,) bool; True flips x for that row (left <-> right hand).
    Returns (N, 63) float32: palm at the origin, divided by a scale reference.
    """
    pts = np.asarray(X, dtype=np.float64).reshape(-1, N_JOINTS, 3).copy()
    pts -= pts[:, :1, :]                                   # palm center -> (0, 0, 0)

    if mirror is not None:
        mirror = np.asarray(mirror, dtype=bool).reshape(-1)
        pts[mirror, :, 0] *= -1.0                          # reflect across the YZ plane

    max_dist = np.linalg.norm(pts, axis=2).max(axis=1)
    if scale == "middle_mcp":
        ref = np.linalg.norm(pts[:, MIDDLE_MCP_IDX], axis=1)
        ref = np.where(ref > 0, ref, max_dist)             # fall back if that joint is missing
    elif scale == "max_dist":
        ref = max_dist
    else:
        raise ValueError(f"unknown scale mode: {scale!r}")
    ref = np.where(ref > 0, ref, 1.0)

    pts /= ref[:, None, None]
    return pts.reshape(-1, FEATURE_DIM).astype(np.float32)


def normalize_points(
    center,
    joints,
    mirror: bool = False,
    scale: str = DEFAULT_SCALE,
) -> np.ndarray:
    """
    Single-frame version for live inference.
    center: (3,) palm center.  joints: (20, 3) in feature order.
    """
    raw = np.vstack([
        np.asarray(center, dtype=np.float64).reshape(1, 3),
        np.asarray(joints, dtype=np.float64).reshape(N_JOINTS - 1, 3),
    ]).reshape(1, FEATURE_DIM)
    return normalize_batch(raw, mirror=np.array([mirror]), scale=scale)[0]
