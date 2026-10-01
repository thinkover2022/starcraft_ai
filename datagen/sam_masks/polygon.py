"""마스크 → 다각형 변환 (객체 인식 모델 분할 학습 형식용)."""

from __future__ import annotations

import cv2
import numpy as np


def _contours(mask: np.ndarray, epsilon: float) -> list[np.ndarray]:
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in cs:
        a = cv2.approxPolyDP(c, epsilon, True).reshape(-1, 2) if epsilon > 0 else c.reshape(-1, 2)
        if len(a) >= 3:
            out.append(a.astype(np.float32))
    return sorted(out, key=lambda p: -abs(cv2.contourArea(p)))


def _bridge(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """두 다각형을 가장 가까운 꼭짓점끼리 가는 선으로 이어 하나의 꼭짓점 열로 만든다."""
    d = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    i, j = np.unravel_index(np.argmin(d), d.shape)
    b_rot = np.roll(b, -j, axis=0)
    return np.concatenate([a[: i + 1], b_rot, b_rot[:1], a[i:]])


def mask_to_polygon(mask: np.ndarray, epsilon: float = 1.0,
                    merge_ratio: float = 0.3) -> np.ndarray | None:
    """가장 큰 조각의 외곽선 (K,2) 픽셀 좌표. 큰 조각 대비 `merge_ratio` 이상인 조각은 이어 붙인다.

    마스크가 비었거나 꼭짓점이 3개 미만이면 None.
    """
    polys = _contours(mask, epsilon)
    if not polys:
        return None
    main = polys[0]
    main_area = abs(cv2.contourArea(main))
    for p in polys[1:]:
        if main_area > 0 and abs(cv2.contourArea(p)) / main_area >= merge_ratio:
            main = _bridge(main, p)
    return main


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """마스크에서 다시 계산한 박스 (x0, y0, x1, y1), 끝 좌표 포함하지 않음."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def polygon_to_yolo(poly: np.ndarray, width: int, height: int) -> list[float]:
    p = poly.astype(np.float64).copy()
    p[:, 0] = np.clip(p[:, 0] / width, 0, 1)
    p[:, 1] = np.clip(p[:, 1] / height, 0, 1)
    return p.reshape(-1).tolist()
