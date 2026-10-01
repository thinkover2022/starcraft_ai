"""맵 좌표 ↔ 화면 좌표, 분할 모델용 사각형 힌트, 그리는 순서."""

from __future__ import annotations

import numpy as np

from .classes import Category, class_for_unit
from .config import MaskConfig
from .frame import FrameRecord, UnitRecord


def to_screen(frame: FrameRecord, x: float, y: float) -> tuple[float, float]:
    """게임은 확대·축소가 없으므로 화면 좌표 = 맵 좌표 − 카메라 위치."""
    return x - frame.camera[0], y - frame.camera[1]


def unit_category(u: UnitRecord) -> Category:
    c = class_for_unit(u.type, u.lifted)
    if c is not None:
        return c.category
    if u.lifted:
        return Category.LIFTED
    return Category.AIR if u.flying else Category.GROUND


def prompt_box(frame: FrameRecord, u: UnitRecord, cfg: MaskConfig) -> np.ndarray | None:
    """충돌 사각형을 화면 좌표로 바꾸고 여유 폭만큼 넓힌 사각형 (x0, y0, x1, y1).

    화면 밖이면 None. 화면 경계에서 잘라낸다.
    """
    l, t, r, b = u.box
    ml, mt, mr, mb = cfg.type_margins.get(u.type) or cfg.margins[unit_category(u).value]
    x0, y0 = to_screen(frame, l - ml, t - mt)
    x1, y1 = to_screen(frame, r + mr, b + mb)
    w, h = frame.screen
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w - 1), min(y1, h - 1)
    if x1 <= x0 or y1 <= y0:
        return None
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def shadow_point(frame: FrameRecord, u: UnitRecord, cfg: MaskConfig) -> tuple[float, float] | None:
    """공중 물체의 그림자 위치(화면 좌표). 그림자 오프셋을 측정하기 전이거나 지상 물체면 None."""
    if cfg.shadow_offset is None or unit_category(u) not in (Category.AIR, Category.LIFTED):
        return None
    sx, sy = to_screen(frame, u.x + cfg.shadow_offset[0], u.y + cfg.shadow_offset[1])
    w, h = frame.screen
    if 0 <= sx < w and 0 <= sy < h:
        return sx, sy
    return None


def draw_order_key(u: UnitRecord) -> tuple[int, int]:
    """게임이 그리는 순서. 값이 클수록 위에 그려진다.

    잠복 물체 < 지상(건물 포함) < 공중, 같은 층에서는 아래쪽(y가 큰) 물체가 위.
    """
    if u.burrowed:
        layer = -1
    elif unit_category(u) in (Category.AIR, Category.LIFTED):
        layer = 1
    else:
        layer = 0
    return layer, u.box[3]


def play_area_mask(shape: tuple[int, int], cfg: MaskConfig) -> np.ndarray | None:
    """게임 화면(조작판 제외) 영역 = True. 설정이 없으면 None(제한 없음)."""
    h, w = shape
    if cfg.console_mask_png:
        import cv2

        from .imageio import imread
        m = imread(cfg.console_mask_png, cv2.IMREAD_GRAYSCALE)
        if m.shape != (h, w):
            raise ValueError(f"조작판 마스크 {cfg.console_mask_png} 크기가 화면({w}x{h})과 다름")
        return m > 127
    if cfg.play_area_bottom is not None:
        m = np.zeros((h, w), dtype=bool)
        m[: cfg.play_area_bottom] = True
        return m
    return None
