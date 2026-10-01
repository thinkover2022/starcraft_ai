"""마스크 품질 자동 검사 (문서 1.4절 [다])."""

from __future__ import annotations

import numpy as np

from ..common.config import MaskConfig

# 불합격 사유 코드
LOW_SCORE = "low_score"            # 분할 모델 신뢰 점수 미달
NO_CENTER = "no_center"            # 마스크가 유닛 중심점을 포함하지 않음 → 엉뚱한 물체
AREA_RATIO = "area_ratio"          # 마스크 넓이 ÷ 충돌 사각형 넓이가 범위 밖
TOO_SMALL = "too_small"


def check_raw_mask(mask: np.ndarray, score: float, center: tuple[float, float],
                   collision_area: float, cfg: MaskConfig) -> list[str]:
    """겹침 정리 전의 마스크를 검사한다. 비어 있으면 합격."""
    flags = []
    if score < cfg.min_score:
        flags.append(LOW_SCORE)
    h, w = mask.shape
    cx, cy = int(round(center[0])), int(round(center[1]))
    if 0 <= cx < w and 0 <= cy < h:
        # 중심점 주변 3x3 안에 마스크가 있으면 통과 (1–2픽셀 어긋남 허용)
        if not mask[max(cy - 1, 0): cy + 2, max(cx - 1, 0): cx + 2].any():
            flags.append(NO_CENTER)
    area = float(mask.sum())
    if area < cfg.min_area_px:
        flags.append(TOO_SMALL)
    elif collision_area > 0:
        lo, hi = cfg.area_ratio_range
        if not lo <= area / collision_area <= hi:
            flags.append(AREA_RATIO)
    return flags
