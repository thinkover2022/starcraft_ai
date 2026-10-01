"""평가 세트 분할 (문서 1.6절). 화면 단위가 아니라 리플레이 단위로 나눈다.

    train / val(①) / test(②) / unseen_maps(③) / live(④)
⑤ 정밀 검수 세트는 사람이 고친 주석으로 따로 만든다 (gold).
"""

from __future__ import annotations

import hashlib

from ..common.config import SplitConfig


def _unit_interval(key: str, seed: str) -> float:
    h = hashlib.sha1(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def assign_split(replay_id: str, map_name: str, source: str, cfg: SplitConfig) -> str:
    """같은 리플레이는 항상 같은 세트로 간다 (실행할 때마다 결과가 같음)."""
    if source == "live":
        return "live"
    held = {m.lower() for m in cfg.held_out_maps}
    if map_name and map_name.lower() in held:
        return "unseen_maps"
    r = _unit_interval(replay_id, cfg.seed)
    if r < cfg.test_fraction:
        return "test"
    if r < cfg.test_fraction + cfg.val_fraction:
        return "val"
    return "train"
