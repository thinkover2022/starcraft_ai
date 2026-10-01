"""사람 검수용 겹침 이미지 생성 (문서 1.4절 [다] 사람 검수).

    python -m datagen.review.overlay --raw data/raw --ann data/ann --out data/review --fraction 0.01

화면 원본(왼쪽)과 마스크·클래스 이름을 겹쳐 그린 화면(오른쪽)을 나란히 저장한다.
불합격 물체는 빨간 사각형, 학습 제외 물체는 표시하지 않는다.
결과 폴더를 컴퓨터 비전 라벨링 도구(CVAT)에 올리거나 이미지 뷰어로 넘겨 보며 불량률을 센다.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import cv2
import numpy as np

from ..common.classes import class_names
from ..common.imageio import imread, imwrite, index_pngs
from ..sam_masks.pipeline import load_ann


def _color(cid: int) -> tuple[int, int, int]:
    h = hashlib.md5(str(cid).encode()).digest()
    return int(h[0]), int(h[1]), int(h[2])


def render(img_bgr: np.ndarray, ann, scale: int = 2) -> np.ndarray:
    names = class_names()
    over = img_bgr.copy()
    for o in ann.objects:
        poly = np.asarray(o.polygon, np.int32)
        col = _color(o.class_id)
        layer = over.copy()
        cv2.fillPoly(layer, [poly], col)
        over = cv2.addWeighted(layer, 0.4, over, 0.6, 0)
        cv2.polylines(over, [poly], True, col, 1)
    over = cv2.resize(over, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    for o in ann.objects:
        x0, y0 = o.bbox[0] * scale, o.bbox[1] * scale
        label = names[o.class_id].replace("Terran_", "")
        cv2.putText(over, label, (x0, max(y0 - 3, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1, cv2.LINE_AA)
    left = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    return np.hstack([left, over])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, type=Path)
    ap.add_argument("--ann", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--fraction", type=float, default=0.01)
    ap.add_argument("--seed", default="review")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    n = 0
    pngs = index_pngs(args.raw)
    for p in sorted(args.ann.glob("*.json")):
        if p.name.startswith("_"):
            continue
        h = int.from_bytes(hashlib.sha1(f"{args.seed}:{p.stem}".encode()).digest()[:8], "big")
        if h / 2**64 >= args.fraction:
            continue
        ann = load_ann(p)
        png = pngs.get(ann.stem)
        if png is None:
            continue
        imwrite(args.out / f"{ann.stem}.png", render(imread(png), ann))
        n += 1
    print(f"검수 이미지 {n}장 → {args.out}")


if __name__ == "__main__":
    main()
