"""데이터 생성 설정 (configs/datagen.yaml)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .classes import Category


@dataclass
class MaskConfig:
    upscale: int = 2                    # 범용 분할 모델에 넣기 전 화면 확대 배율
    # 분류별 기본 여유 폭 (왼, 위, 오른, 아래), 픽셀. 충돌 사각형 → 그림 범위 힌트
    margins: dict[str, tuple[int, int, int, int]] = field(default_factory=lambda: {
        Category.GROUND.value: (4, 6, 4, 4),
        Category.AIR.value: (6, 8, 6, 6),
        Category.BUILDING.value: (8, 32, 8, 8),
        Category.LIFTED.value: (8, 16, 8, 8),
        Category.RESOURCE.value: (4, 8, 4, 4),
    })
    type_margins: dict[str, tuple[int, int, int, int]] = field(default_factory=dict)  # 종류별 덮어쓰기
    # 공중 물체 그림자 위치 = 유닛 중심 + 이 값. 측정 전에는 None(그림자 음성 점 사용 안 함)
    shadow_offset: tuple[int, int] | None = None
    min_visible_fraction: float = 0.2   # 겹침 정리 후 원래 넓이 대비 이 비율 미만이면 "가려짐"
    min_area_px: int = 12
    min_score: float = 0.7
    area_ratio_range: tuple[float, float] = (0.3, 4.0)  # 마스크 넓이 ÷ 충돌 사각형 넓이 허용 범위
    max_bad_fraction: float = 0.3       # 한 화면에서 불합격 비율이 이보다 크면 화면 전체 버림
    # 화면 하단 조작판에 가려지는 영역. 조작판 모양 마스크 PNG(255=게임 화면, 0=조작판) 경로
    console_mask_png: str | None = None
    # 위 PNG가 없을 때 쓰는 간단한 대안: 이 y좌표 아래는 모두 조작판으로 간주. 측정 전에는 None
    play_area_bottom: int | None = None


@dataclass
class SplitConfig:
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    held_out_maps: list[str] = field(default_factory=list)  # 평가 세트 ③ 처음 보는 맵
    seed: str = "sc-terran-v1"


@dataclass
class DatagenConfig:
    mask: MaskConfig = field(default_factory=MaskConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    min_instances_per_class: int = 3000  # 드문 클래스 반복 표본 목표


def load_config(path: str | Path | None) -> DatagenConfig:
    cfg = DatagenConfig()
    if path is None:
        return cfg
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for k, v in (d.get("mask") or {}).items():
        if not hasattr(cfg.mask, k):
            raise KeyError(f"알 수 없는 mask 설정: {k}")
        if k in ("margins", "type_margins"):
            v = {name: tuple(m) for name, m in v.items()}
            v = {**getattr(cfg.mask, k), **v}
        elif isinstance(v, list):
            v = tuple(v)
        setattr(cfg.mask, k, v)
    for k, v in (d.get("split") or {}).items():
        if not hasattr(cfg.split, k):
            raise KeyError(f"알 수 없는 split 설정: {k}")
        setattr(cfg.split, k, v)
    if "min_instances_per_class" in d:
        cfg.min_instances_per_class = d["min_instances_per_class"]
    return cfg
