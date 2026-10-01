"""시험용 가짜 화면·프레임 기록·분할기."""

from __future__ import annotations

import json

import numpy as np

from datagen.common.frame import FrameRecord, UnitRecord

BG = (40, 60, 40)


def unit(uid, type_, x, y, half=(8, 8), owner=0, color=111, **kw) -> UnitRecord:
    hw, hh = half
    return UnitRecord(id=uid, type=type_, owner=owner, color=color, x=x, y=y,
                      box=(x - hw, y - hh, x + hw - 1, y + hh - 1), **kw)


def frame(units, camera=(1000, 2000), **kw) -> FrameRecord:
    return FrameRecord(replay_id=kw.pop("replay_id", "r0000abcd"), frame=kw.pop("frame", 120),
                       camera=camera, units=units, map_name=kw.pop("map_name", "Fighting Spirit"),
                       **kw)


def draw(fr: FrameRecord, colors: dict[int, tuple[int, int, int]] | None = None,
         sprite_scale: float = 1.0) -> np.ndarray:
    """유닛을 단색 사각형으로 그린 화면. 그리는 순서는 게임과 같게(아래쪽·공중이 위)."""
    from datagen.common.geometry import draw_order_key
    img = np.zeros((fr.screen[1], fr.screen[0], 3), np.uint8)
    img[:] = BG
    for u in sorted(fr.units, key=draw_order_key):
        l, t, r, b = u.box
        cx, cy = (l + r) / 2, (t + b) / 2
        hw, hh = (r - l + 1) / 2 * sprite_scale, (b - t + 1) / 2 * sprite_scale
        x0, y0 = int(cx - hw - fr.camera[0]), int(cy - hh - fr.camera[1])
        x1, y1 = int(cx + hw - fr.camera[0]), int(cy + hh - fr.camera[1])
        col = (colors or {}).get(u.id, (200, 30 + (u.id * 37) % 200, 30))
        img[max(y0, 0): max(y1, 0), max(x0, 0): max(x1, 0)] = col
    return img


class BlobSegmenter:
    """사각형 안에서 배경색이 아닌 픽셀 중 사각형 중심의 색과 같은 픽셀을 마스크로 낸다."""

    def segment(self, image_rgb, boxes, neg_points=None):
        h, w = image_rgb.shape[:2]
        masks = np.zeros((len(boxes), h, w), bool)
        for i, (x0, y0, x1, y1) in enumerate(np.asarray(boxes).round().astype(int)):
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            col = image_rgb[cy, cx]
            region = np.all(image_rgb[y0: y1 + 1, x0: x1 + 1] == col, axis=-1)
            masks[i, y0: y1 + 1, x0: x1 + 1] = region & np.any(col != BG)
        return masks, np.full(len(boxes), 0.95, np.float32)


# 라벨 수집 모듈(C++)이 쓰는 JSON과 같은 모양의 예시 — 형식 계약 시험용
CPP_SAMPLE = json.dumps({
    "version": 1, "replay_id": "r1a2b3c4d", "frame": 2400, "map": "Fighting Spirit",
    "source": "replay", "screen": [640, 480], "capture_lag_frames": 1, "camera": [1200, 900],
    "vision_players": [0, 1],
    "resources": [{"player": 0, "minerals": 350, "gas": 100, "supply_used": 38, "supply_total": 54}],
    "units": [{"id": 17, "type": "Terran_Marine", "owner": 0, "color": 111, "x": 1500, "y": 1100,
               "box": [1491, 1090, 1508, 1109], "prev": [[1498, 1100], [1496, 1100]], "hp": 40,
               "selected": False, "flying": False, "lifted": False, "constructing": False,
               "cloaked": False, "burrowed": False}],
    "fogged_buildings": 0,
})
