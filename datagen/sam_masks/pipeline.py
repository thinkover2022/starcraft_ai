"""화면 1장 처리: 사각형 힌트 → 범용 분할 모델 → 품질 검사 → 겹침 정리 → 조작판 제거 → 다각형.

결과는 화면당 주석 JSON 하나로 저장한다 (annotations/{stem}.json).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..common.classes import class_for_unit, is_visual
from ..common.config import MaskConfig
from ..common.frame import FrameRecord
from ..common.geometry import draw_order_key, play_area_mask, prompt_box, shadow_point, to_screen
from .overlap import resolve_overlap
from .polygon import mask_bbox, mask_to_polygon
from .quality import check_raw_mask
from .segmenter import BoxSegmenter

# 학습에서 빼는 사유 (불량은 아니지만 정답으로 쓰지 않음)
CLOAKED = "cloaked"
OCCLUDED = "occluded"
OFF_PLAY_AREA = "off_play_area"
NO_POLYGON = "no_polygon"


@dataclass
class ObjectAnn:
    unit_id: int
    type: str
    class_id: int
    polygon: list[list[float]]          # 픽셀 좌표 [[x, y], ...]
    bbox: tuple[int, int, int, int]     # 마스크에서 다시 계산한 박스
    score: float
    visible_fraction: float
    owner: int
    color: int
    hp: int
    selected: bool
    constructing: bool


@dataclass
class FrameAnn:
    stem: str
    replay_id: str
    frame: int
    map_name: str
    source: str
    width: int
    height: int
    objects: list[ObjectAnn] = field(default_factory=list)
    ignored: list[dict] = field(default_factory=list)   # 학습 제외 (사유 포함)
    rejected_objects: list[dict] = field(default_factory=list)  # 품질 불합격
    frame_rejected: bool = False
    reject_reason: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def load_ann(path: str | Path) -> FrameAnn:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    d["objects"] = [ObjectAnn(**o) for o in d["objects"]]
    return FrameAnn(**d)


def process_frame(image_rgb: np.ndarray, frame: FrameRecord, segmenter: BoxSegmenter,
                  cfg: MaskConfig) -> FrameAnn:
    h, w = image_rgb.shape[:2]
    if (w, h) != tuple(frame.screen):
        raise ValueError(f"{frame.stem}: 화면 크기 {w}x{h}가 기록({frame.screen})과 다름")
    ann = FrameAnn(frame.stem, frame.replay_id, frame.frame, frame.map_name, frame.source, w, h)
    if frame.fogged_buildings > 0:
        ann.frame_rejected, ann.reject_reason = True, "fogged_buildings"
        return ann

    # 1. 그려지는 물체 전부에 사각형 힌트를 만든다. 대상 클래스가 아닌 물체도 가림 판단에는 쓴다.
    units, boxes, negs = [], [], []
    for u in frame.units:
        if not is_visual(u.type):
            continue
        box = prompt_box(frame, u, cfg)
        if box is None:
            continue
        units.append(u)
        boxes.append(box)
        negs.append(shadow_point(frame, u, cfg))
    if not units:
        return ann

    # 2. 범용 분할 모델
    masks, scores = segmenter.segment(image_rgb, np.stack(boxes), negs)

    # 3. 겹침 정리 전 품질 검사
    raw_flags = []
    for u, m, s in zip(units, masks, scores):
        l, t, r, b = u.box
        raw_flags.append(check_raw_mask(m, float(s), to_screen(frame, u.x, u.y),
                                        float((r - l + 1) * (b - t + 1)), cfg))

    # 4. 겹침 정리: 불합격 마스크는 다른 물체를 가리지 않게 빼고 정리한다.
    keep = np.array([not f for f in raw_flags])
    work = masks.copy()
    work[~keep] = False
    resolved, frac = resolve_overlap(work, [draw_order_key(u) for u in units])

    # 5. 조작판 영역 제거
    play = play_area_mask((h, w), cfg)
    if play is not None:
        resolved &= play[None]

    # 6. 정답 객체로 정리
    n_target = n_bad = 0
    for i, u in enumerate(units):
        cls = class_for_unit(u.type, u.lifted)
        if cls is None:
            continue                      # 이번 단계 대상 클래스가 아님 (가림 판단에만 사용)
        n_target += 1
        base = {"unit_id": u.id, "type": u.type}
        if raw_flags[i]:
            n_bad += 1
            ann.rejected_objects.append({**base, "flags": raw_flags[i], "score": float(scores[i])})
            continue
        if u.cloaked:
            ann.ignored.append({**base, "reason": CLOAKED})
            continue
        if frac[i] < cfg.min_visible_fraction:
            ann.ignored.append({**base, "reason": OCCLUDED, "visible_fraction": float(frac[i])})
            continue
        m = resolved[i]
        bbox = mask_bbox(m)
        if bbox is None:
            ann.ignored.append({**base, "reason": OFF_PLAY_AREA})
            continue
        poly = mask_to_polygon(m)
        if poly is None:
            ann.ignored.append({**base, "reason": NO_POLYGON})
            continue
        ann.objects.append(ObjectAnn(
            unit_id=u.id, type=u.type, class_id=cls.id, polygon=poly.round(1).tolist(),
            bbox=bbox, score=float(scores[i]), visible_fraction=float(frac[i]),
            owner=u.owner, color=u.color, hp=u.hp, selected=u.selected,
            constructing=u.constructing,
        ))

    if n_target and n_bad / n_target > cfg.max_bad_fraction:
        ann.frame_rejected, ann.reject_reason = True, "too_many_bad_masks"
    return ann
