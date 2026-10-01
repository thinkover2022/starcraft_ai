"""겹친 마스크 정리: 같은 픽셀을 여러 물체가 차지하면 게임이 위에 그리는 물체에게 준다."""

from __future__ import annotations

import numpy as np


def resolve_overlap(masks: np.ndarray, order_keys: list) -> tuple[np.ndarray, np.ndarray]:
    """masks (N,H,W) bool, order_keys 길이 N (클수록 위에 그려짐).

    반환: 정리된 masks (N,H,W), 남은 넓이 비율 (N,) — 원래 넓이가 0이면 0.
    """
    n = len(masks)
    if n == 0:
        return masks, np.zeros((0,), np.float32)
    label = np.full(masks.shape[1:], -1, np.int32)
    # 아래에 그려지는 것부터 칠하면 위에 그려지는 것이 덮어쓴다.
    for i in sorted(range(n), key=lambda i: order_keys[i]):
        label[masks[i]] = i
    out = np.stack([label == i for i in range(n)])
    before = masks.reshape(n, -1).sum(1).astype(np.float32)
    after = out.reshape(n, -1).sum(1).astype(np.float32)
    frac = np.divide(after, before, out=np.zeros_like(after), where=before > 0)
    return out, frac
