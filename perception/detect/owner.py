"""소유자 판정 (문서 1.8절): 마스크 안의 팀 색상 픽셀을 세어 플레이어 색상을 정한다.

팀 색상 RGB 값을 손으로 적지 않고, 자동 라벨 데이터에서 직접 뽑는다(보정).
"색상 c 플레이어의 유닛 안에서는 자주 나오고, 다른 색 유닛 안에서는 거의 안 나오는 RGB"가
색상 c의 팀 색상이다.

    # 보정: 학습 세트 주석으로 색상표 만들기
    python -m perception.detect.owner calibrate --raw data/raw --ann data/ann --out configs/team_palette.json
    # 평가: 정답 소유자 색과 비교
    python -m perception.detect.owner evaluate --raw data/raw --ann data/ann --palette configs/team_palette.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


def pack_rgb(pixels: np.ndarray) -> np.ndarray:
    p = pixels.reshape(-1, 3).astype(np.uint32)
    return (p[:, 0] << 16) | (p[:, 1] << 8) | p[:, 2]


def polygon_mask(polygon, shape) -> np.ndarray:
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [np.asarray(polygon, np.int32)], 1)
    return m.astype(bool)


class TeamPalette:
    def __init__(self, colors: dict[int, set[int]]):
        self.colors = colors                      # 색상 번호 → 묶음 RGB 값 집합
        self._lut = {rgb: c for c, s in colors.items() for rgb in s}

    @classmethod
    def load(cls, path) -> "TeamPalette":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls({int(c): set(v) for c, v in d["colors"].items()})

    def save(self, path, meta: dict | None = None):
        Path(path).write_text(json.dumps(
            {"colors": {str(c): sorted(s) for c, s in self.colors.items()}, "meta": meta or {}},
            indent=1), encoding="utf-8")

    def classify(self, image_rgb: np.ndarray, mask: np.ndarray) -> tuple[int | None, float]:
        """(색상 번호, 신뢰도). 팀 색상 픽셀이 하나도 없으면 (None, 0)."""
        votes = Counter(self._lut[v] for v in pack_rgb(image_rgb[mask]).tolist() if v in self._lut)
        if not votes:
            return None, 0.0
        c, n = votes.most_common(1)[0]
        return c, n / sum(votes.values())


def calibrate(samples, min_purity: float = 0.95, min_count: int = 50,
              max_per_color: int = 32) -> TeamPalette:
    """samples: (image_rgb, mask, color_id) 반복자."""
    per_color: dict[int, Counter] = defaultdict(Counter)
    total: Counter = Counter()
    for img, mask, color in samples:
        vals = pack_rgb(img[mask]).tolist()
        per_color[color].update(vals)
        total.update(vals)
    out = {}
    for c, cnt in per_color.items():
        cand = [(v, n) for v, n in cnt.items() if n >= min_count and n / total[v] >= min_purity]
        cand.sort(key=lambda x: -x[1])
        out[c] = {v for v, _ in cand[:max_per_color]}
    return TeamPalette(out)


def _samples(raw: Path, ann_dir: Path, limit: int | None = None):
    from datagen.common.imageio import imread_rgb, index_pngs
    from datagen.sam_masks.pipeline import load_ann
    pngs = index_pngs(raw)
    n = 0
    for p in sorted(ann_dir.glob("*.json")):
        if p.name.startswith("_"):
            continue
        ann = load_ann(p)
        if ann.frame_rejected or not ann.objects:
            continue
        png = pngs.get(ann.stem)
        if png is None:
            continue
        img = imread_rgb(png)
        for o in ann.objects:
            if o.type.startswith("Resource_"):
                continue
            yield img, polygon_mask(o.polygon, img.shape), o.color
        n += 1
        if limit and n >= limit:
            return


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate")
    c.add_argument("--raw", required=True, type=Path)
    c.add_argument("--ann", required=True, type=Path)
    c.add_argument("--out", required=True, type=Path)
    c.add_argument("--limit", type=int, default=5000, help="사용할 화면 수 상한")
    e = sub.add_parser("evaluate")
    e.add_argument("--raw", required=True, type=Path)
    e.add_argument("--ann", required=True, type=Path)
    e.add_argument("--palette", required=True, type=Path)
    e.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    if args.cmd == "calibrate":
        pal = calibrate(_samples(args.raw, args.ann, args.limit))
        pal.save(args.out, {"frames": args.limit})
        for col, s in sorted(pal.colors.items()):
            print(f"색상 {col}: 팀 색상 {len(s)}개")
        return
    pal = TeamPalette.load(args.palette)
    ok = n = none = 0
    for img, mask, color in _samples(args.raw, args.ann, args.limit):
        pred, _ = pal.classify(img, mask)
        n += 1
        none += pred is None
        ok += pred == color
    print(json.dumps({"objects": n, "accuracy": ok / max(n, 1), "no_team_pixels": none / max(n, 1)},
                     indent=2))


if __name__ == "__main__":
    main()
