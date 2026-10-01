"""마스크 생성 프로그램 실행 진입점.

    python -m datagen.sam_masks.run --raw data/raw --out data/ann --config configs/datagen.yaml

--segmenter box 를 주면 범용 분할 모델 대신 사각형 힌트를 그대로 마스크로 쓴다
(그래픽 카드 없이 파이프라인 전체를 점검하는 용도. 학습 데이터로 쓰면 안 됨).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from ..common.config import load_config
from ..common.frame import iter_frames
from ..common.imageio import imread_rgb
from .pipeline import process_frame


class BoxAsMaskSegmenter:
    """점검용: 사각형 = 마스크."""

    def segment(self, image_rgb, boxes, neg_points=None):
        h, w = image_rgb.shape[:2]
        masks = np.zeros((len(boxes), h, w), bool)
        for i, (x0, y0, x1, y1) in enumerate(np.asarray(boxes).round().astype(int)):
            masks[i, y0: y1 + 1, x0: x1 + 1] = True
        return masks, np.ones(len(boxes), np.float32)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, help="PNG + JSON 원본 폴더")
    ap.add_argument("--out", required=True, help="주석 JSON 저장 폴더")
    ap.add_argument("--config", default=None)
    ap.add_argument("--segmenter", choices=["sam2", "box"], default="sam2")
    ap.add_argument("--model", default="facebook/sam2.1-hiera-large")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.segmenter == "sam2":
        from .segmenter import Sam2BoxSegmenter
        seg = Sam2BoxSegmenter(args.model, upscale=cfg.mask.upscale)
    else:
        seg = BoxAsMaskSegmenter()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    for png, frame in iter_frames(args.raw):
        dst = out / f"{frame.stem}.json"
        if args.skip_existing and dst.exists():
            continue
        img = imread_rgb(png)
        ann = process_frame(img, frame, seg, cfg.mask)
        dst.write_text(ann.to_json(), encoding="utf-8")
        stats["frames"] += 1
        stats["frames_rejected"] += ann.frame_rejected
        if ann.reject_reason:
            stats["reject_" + ann.reject_reason] += 1
        stats["objects"] += len(ann.objects)
        stats["rejected_objects"] += len(ann.rejected_objects)
        for ig in ann.ignored:
            stats["ignored_" + ig["reason"]] += 1
        for r in ann.rejected_objects:
            for f in r["flags"]:
                stats["flag_" + f] += 1
        if stats["frames"] % 500 == 0:
            print(dict(stats), flush=True)
    print(json.dumps(dict(stats), indent=2, ensure_ascii=False))
    (out / "_stats.json").write_text(json.dumps(dict(stats), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
