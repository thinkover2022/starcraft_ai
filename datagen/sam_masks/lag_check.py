"""화면-상태 어긋남 실험 (문서 1.4절 [나], 1단계 2주차).

게임을 멈춘 순간의 화면이 기록한 프레임과 같은지, 1–2프레임 전인지 확인한다.
움직이는 유닛에 대해 "마스크 중심"과 "k프레임 전 위치"의 거리를 k = 0, 1, 2 별로 재고,
중앙값이 가장 작은 k를 어긋남 값으로 고른다. 결과를 capture_server.py --lag 에 넣는다.

    python -m datagen.sam_masks.lag_check --raw data/pilot_raw --ann data/pilot_ann
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..common.frame import iter_frames
from .pipeline import load_ann


def candidate_positions(u) -> list[tuple[int, int]]:
    return [(u.x, u.y)] + [tuple(p) for p in u.prev[:2]]


def lag_errors(frames_and_anns, min_move: float = 2.0) -> dict[int, list[float]]:
    errs: dict[int, list[float]] = {0: [], 1: [], 2: []}
    for frame, ann in frames_and_anns:
        units = {u.id: u for u in frame.units}
        for o in ann.objects:
            u = units.get(o.unit_id)
            if u is None or len(u.prev) < 2 or u.flying or o.visible_fraction < 0.95:
                continue
            cands = candidate_positions(u)
            if np.hypot(cands[0][0] - cands[2][0], cands[0][1] - cands[2][1]) < min_move:
                continue                                  # 움직이지 않는 유닛은 구분에 도움이 안 됨
            poly = np.asarray(o.polygon)
            cx, cy = poly[:, 0].mean(), poly[:, 1].mean()
            for k, (x, y) in enumerate(cands):
                sx, sy = x - frame.camera[0], y - frame.camera[1]
                errs[k].append(float(np.hypot(cx - sx, cy - sy)))
    return errs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, type=Path)
    ap.add_argument("--ann", required=True, type=Path)
    args = ap.parse_args(argv)

    def pairs():
        for _, frame in iter_frames(args.raw):
            p = args.ann / f"{frame.stem}.json"
            if p.exists():
                yield frame, load_ann(p)

    errs = lag_errors(pairs())
    summary = {k: {"units": len(v), "median_px": float(np.median(v)) if v else None}
               for k, v in errs.items()}
    best = min((k for k in summary if summary[k]["median_px"] is not None),
               key=lambda k: summary[k]["median_px"], default=None)
    print(json.dumps({"by_lag": summary, "best_lag_frames": best}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
