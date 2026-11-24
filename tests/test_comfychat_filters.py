#!/usr/bin/env python

"""Tests for `comfychat_filters` package."""

import numpy as np
import torch

from src.comfychat_filters.nodes import (
    ApplyComfychatFilterMHR,
    ApplyComfychatFilterMediaPipe,
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #

def test_node_registration():
    """Both filter nodes are exported under their class names."""
    assert NODE_CLASS_MAPPINGS["ApplyComfychatFilterMHR"] is ApplyComfychatFilterMHR
    assert NODE_CLASS_MAPPINGS["ApplyComfychatFilterMediaPipe"] is ApplyComfychatFilterMediaPipe


def test_display_names():
    assert "ApplyComfychatFilterMHR" in NODE_DISPLAY_NAME_MAPPINGS
    assert "ApplyComfychatFilterMediaPipe" in NODE_DISPLAY_NAME_MAPPINGS


# --------------------------------------------------------------------------- #
# Synthetic FACE_LANDMARKS helpers
#
# A self-consistent face: camera-space 3D points (Z<0, in front of the lens)
# projected through a pinhole model. The MediaPipe camera convention yields a
# negative fx and positive fy, which is exactly what the node must recover.
# --------------------------------------------------------------------------- #

N_LANDMARKS = 478
IMG_W = IMG_H = 512
FX, FY, CX, CY = -417.7, 417.7, 256.0, 256.0


def _project(p_cam):
    u = FX * (p_cam[:, 0] / p_cam[:, 2]) + CX
    v = FY * (p_cam[:, 1] / p_cam[:, 2]) + CY
    return np.stack([u, v], axis=1)


def _canonical():
    l3 = np.zeros((N_LANDMARKS, 3), dtype=np.float64)
    for i in range(N_LANDMARKS):
        l3[i, 0] = (i % 20) * 2.0 - 20.0
        l3[i, 1] = (i // 20) * 2.0 - 20.0
        l3[i, 2] = (i % 7) * 2.0 - 6.0
    l3[473] = (-10.0, 0.0, 0.0)   # left iris centre
    l3[468] = (10.0, 0.0, 0.0)    # right iris centre
    l3[1] = (0.0, -8.0, 5.0)      # nose tip (below the eye line)
    return l3


def _rot_y(deg):
    t = np.radians(deg)
    return np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])


def build_face(yaw_deg=0.0):
    """Return (face_dict, p_cam) with a consistent transformation matrix."""
    l3 = _canonical()
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = _rot_y(yaw_deg)
    M[2, 3] = -100.0
    p_cam = l3 @ M[:3, :3].T + M[:3, 3]
    l2 = _project(p_cam)
    face = {
        "bbox_xyxy": np.array([150, 150, 360, 360], dtype=np.float32),
        "blendshapes": {},
        "landmarks_xy": l2.astype(np.float32),
        "landmarks_3d": l3.astype(np.float32),
        "presence": 1.0,
        "score": 0.9,
        "transformation_matrix": M.astype(np.float32),
    }
    return face, p_cam


def make_inputs(face):
    face_landmarks = {"frames": [[face]], "image_size": (IMG_H, IMG_W), "connection_sets": {}}
    bg = torch.zeros((1, IMG_H, IMG_W, 3), dtype=torch.float32)
    bg[0, :, :, 0] = 0.5  # red background
    filt = torch.zeros((1, 128, 128, 4), dtype=torch.float32)
    filt[0, :, :, 1] = 1.0  # green
    filt[0, :, :, 3] = 1.0  # opaque
    return face_landmarks, bg, filt


def eye_midpoint(face):
    l = face["landmarks_xy"][473]
    r = face["landmarks_xy"][468]
    return ((l[0] + r[0]) / 2.0, (l[1] + r[1]) / 2.0)


def centroid(alpha):
    ys, xs = np.nonzero(alpha > 0.01)
    if len(xs) == 0:
        return None
    return (float(xs.mean()), float(ys.mean()))


def run(node, face, perspective, **overrides):
    fl, bg, filt = make_inputs(face)
    kwargs = dict(
        face_index=0, offset_x=0.0, offset_y=0.0, offset_z=0.1,
        scale_x=1.0, scale_y=1.0, tilt=0.0, rot_offset_x=0.0, rot_offset_y=0.0,
        smoothing_window=1, perspective=perspective, occlusion_culling=True,
        fade_threshold=0.15,
    )
    kwargs.update(overrides)
    return node.apply_perspective_filter(fl, bg, filt, **kwargs)


# --------------------------------------------------------------------------- #
# Node metadata
# --------------------------------------------------------------------------- #

def test_mediapipe_node_metadata():
    node = ApplyComfychatFilterMediaPipe()
    assert node.RETURN_TYPES == ("IMAGE", "IMAGE")
    assert node.RETURN_NAMES == ("IMAGE", "FILTER_ONLY")
    assert node.FUNCTION == "apply_perspective_filter"
    assert node.CATEGORY == "MediaPipe/Effects"


def test_mediapipe_node_inputs():
    inputs = ApplyComfychatFilterMediaPipe.INPUT_TYPES()["required"]
    expected = (
        "face_landmarks", "background_image", "filter_image", "face_index",
        "offset_x", "offset_y", "offset_z", "scale_x", "scale_y", "tilt",
        "rot_offset_x", "rot_offset_y", "smoothing_window", "perspective",
        "occlusion_culling", "fade_threshold",
    )
    for key in expected:
        assert key in inputs, f"missing input {key!r}"
    assert inputs["face_landmarks"][0] == "FACE_LANDMARKS"
    assert inputs["background_image"][0] == "IMAGE"
    assert inputs["filter_image"][0] == "IMAGE"


# --------------------------------------------------------------------------- #
# Static helpers
# --------------------------------------------------------------------------- #

def test_estimate_intrinsics_recovers_known():
    face, p_cam = build_face()
    intr = ApplyComfychatFilterMediaPipe._estimate_intrinsics(
        p_cam.astype(np.float64), face["landmarks_xy"].astype(np.float64)
    )
    assert intr is not None
    fx, fy, cx, cy = intr
    assert abs(fx - FX) < 5.0
    assert abs(fy - FY) < 5.0
    assert abs(cx - CX) < 3.0
    assert abs(cy - CY) < 3.0


def test_estimate_intrinsics_rejects_degenerate():
    # All landmarks collapse to one point -> no recoverable focal length.
    p_cam = np.zeros((20, 3), dtype=np.float64)
    p_cam[:, 2] = -100.0
    l2 = np.zeros((20, 2), dtype=np.float64)
    assert ApplyComfychatFilterMediaPipe._estimate_intrinsics(p_cam, l2) is None


def test_project_2d_returns_quad():
    dst = ApplyComfychatFilterMediaPipe._project_2d(
        np.array([214.0, 256.0]), np.array([298.0, 256.0]),
        0.0, 0.0, 1.0, 1.0, 0.0,
    )
    assert dst is not None
    assert dst.shape == (4, 2)


def test_project_3d_returns_quad():
    dst = ApplyComfychatFilterMediaPipe._project_3d(
        np.array([-10.0, 0.0, -100.0]), np.array([10.0, 0.0, -100.0]),
        np.array([0.0, -8.0, -95.0]), (FX, FY, CX, CY),
        0.0, 0.0, 0.1, 1.0, 1.0, 0.0, 0.0, 0.0,
    )
    assert dst is not None
    assert dst.shape == (4, 2)


# --------------------------------------------------------------------------- #
# End-to-end
# --------------------------------------------------------------------------- #

def test_apply_2d_billboard():
    node = ApplyComfychatFilterMediaPipe()
    face, _ = build_face()
    out_img, out_filt = run(node, face, "2D")
    assert out_img.shape == (1, IMG_H, IMG_W, 3)
    assert out_filt.shape == (1, IMG_H, IMG_W, 4)
    assert out_img.min() >= 0.0 and out_img.max() <= 1.0
    c = centroid(out_filt[0, :, :, 3].numpy())
    mx, my = eye_midpoint(face)
    assert c is not None
    assert abs(c[0] - mx) < 30.0
    assert abs(c[1] - my) < 30.0


def test_apply_3d():
    node = ApplyComfychatFilterMediaPipe()
    face, _ = build_face()
    out_img, out_filt = run(node, face, "3D")
    assert out_img.shape == (1, IMG_H, IMG_W, 3)
    c = centroid(out_filt[0, :, :, 3].numpy())
    mx, my = eye_midpoint(face)
    assert c is not None
    assert abs(c[0] - mx) < 40.0
    assert abs(c[1] - my) < 40.0


def test_apply_3d_follows_yaw():
    """A yawed head must still render the filter (3D, not a silent drop)."""
    node = ApplyComfychatFilterMediaPipe()
    face, _ = build_face(yaw_deg=35.0)
    _, out_filt = run(node, face, "3D")
    c = centroid(out_filt[0, :, :, 3].numpy())
    assert c is not None


def test_apply_no_face_returns_clean_background():
    node = ApplyComfychatFilterMediaPipe()
    empty = {"frames": [[]], "image_size": (IMG_H, IMG_W), "connection_sets": {}}
    bg = torch.zeros((1, IMG_H, IMG_W, 3), dtype=torch.float32)
    bg[0, :, :, 0] = 0.5
    filt = torch.zeros((1, 128, 128, 4), dtype=torch.float32)
    filt[0, :, :, 1] = 1.0
    filt[0, :, :, 3] = 1.0
    out_img, out_filt = node.apply_perspective_filter(
        empty, bg, filt, 0, 0.0, 0.0, 0.1, 1.0, 1.0, 0.0, 0.0, 0.0, 1, "2D", True, 0.15,
    )
    assert float((out_img[0].numpy() - bg[0].numpy()).max()) < 1e-6
    assert float(out_filt[0, :, :, 3].numpy().max()) < 1e-6


def test_apply_adds_alpha_when_missing():
    """A 3-channel filter image gets an opaque alpha channel appended."""
    node = ApplyComfychatFilterMediaPipe()
    face, _ = build_face()
    fl, bg, _ = make_inputs(face)
    filt_rgb = torch.zeros((1, 128, 128, 3), dtype=torch.float32)
    filt_rgb[0, :, :, 1] = 1.0
    out_img, out_filt = node.apply_perspective_filter(
        fl, bg, filt_rgb, 0, 0.0, 0.0, 0.1, 1.0, 1.0, 0.0, 0.0, 0.0, 1, "2D", True, 0.15,
    )
    assert out_filt.shape == (1, IMG_H, IMG_W, 4)
    assert centroid(out_filt[0, :, :, 3].numpy()) is not None
