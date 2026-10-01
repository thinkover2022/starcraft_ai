"""데이터셋 구성 프로그램 (문서 1.4절 [라]).

    python -m datagen.build_dataset.build --raw data/raw --ann data/ann --out data/sc_terran_v1 \
        --config configs/datagen.yaml

만드는 것:
    images/{세트}/*.png, labels/{세트}/*.txt   객체 인식 모델(YOLO) 분할 학습 형식
    train.txt                                   드문 클래스 반복 표본이 반영된 학습 목록
    attrs_{세트}.json                           소유자·색상·체력 등 학습 형식에 못 넣는 속성
    sc_terran.yaml, sc_terran_{평가세트}.yaml   학습·평가용 데이터셋 설정 파일
    report.json                                 세트별 화면 수, 클래스별 객체 수
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

from ..common.classes import class_names
from ..common.config import load_config
from ..common.imageio import index_pngs
from ..sam_masks.pipeline import load_ann
from ..sam_masks.polygon import polygon_to_yolo
from .split import assign_split

EVAL_SPLITS = ["test", "unseen_maps", "live", "gold"]


def yolo_lines(ann) -> list[str]:
    lines = []
    for o in ann.objects:
        coords = polygon_to_yolo(np.asarray(o.polygon), ann.width, ann.height)
        lines.append(f"{o.class_id} " + " ".join(f"{v:.6f}" for v in coords))
    return lines


def repeat_factors(image_classes: dict[str, set[int]], class_counts: Counter,
                   target: int, cap: int = 20) -> dict[str, int]:
    """화면마다 반복 횟수. 드문 클래스가 든 화면일수록 많이 반복한다 (최대 cap)."""
    rc = {c: min(cap, max(1.0, target / n)) for c, n in class_counts.items() if n > 0}
    return {stem: math.ceil(max((rc[c] for c in cs), default=1.0))
            for stem, cs in image_classes.items()}


def _place(src: Path, dst: Path, copy: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def build(raw: Path, ann_dir: Path, out: Path, cfg, gold_dir: Path | None = None,
          max_rejected_per_frame: int = 0, copy: bool = False) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    report = {"frames": Counter(), "skipped": Counter(), "classes": defaultdict(Counter)}
    attrs = defaultdict(dict)
    train_classes: dict[str, set[int]] = {}
    train_counts: Counter = Counter()

    sources = [(p, None) for p in sorted(ann_dir.glob("*.json")) if not p.name.startswith("_")]
    if gold_dir is not None:
        sources += [(p, "gold") for p in sorted(gold_dir.glob("*.json")) if not p.name.startswith("_")]

    pngs = index_pngs(raw)
    for path, forced in sources:
        ann = load_ann(path)
        if ann.frame_rejected:
            report["skipped"]["frame_rejected"] += 1
            continue
        if len(ann.rejected_objects) > max_rejected_per_frame:
            # 라벨이 빠진 물체가 있으면 모델이 그것을 '배경'으로 잘못 배우므로 화면째 뺀다.
            report["skipped"]["rejected_objects"] += 1
            continue
        split = forced or assign_split(ann.replay_id, ann.map_name, ann.source, cfg.split)
        png = pngs.get(ann.stem)
        if png is None:
            report["skipped"]["missing_png"] += 1
            continue
        _place(png, out / "images" / split / png.name, copy)
        lbl = out / "labels" / split / f"{ann.stem}.txt"
        lbl.parent.mkdir(parents=True, exist_ok=True)
        lbl.write_text("\n".join(yolo_lines(ann)) + ("\n" if ann.objects else ""), encoding="utf-8")
        attrs[split][ann.stem] = [
            {"class_id": o.class_id, "bbox": o.bbox, "owner": o.owner, "color": o.color,
             "hp": o.hp, "selected": o.selected, "constructing": o.constructing}
            for o in ann.objects
        ]
        report["frames"][split] += 1
        for o in ann.objects:
            report["classes"][split][o.class_id] += 1
        if split == "train":
            cs = {o.class_id for o in ann.objects}
            train_classes[ann.stem] = cs
            train_counts.update(o.class_id for o in ann.objects)

    reps = repeat_factors(train_classes, train_counts, cfg.min_instances_per_class)
    with open(out / "train.txt", "w", encoding="utf-8") as f:
        for stem, r in sorted(reps.items()):
            f.write(f"./images/train/{stem}.png\n" * r)

    for split, d in attrs.items():
        (out / f"attrs_{split}.json").write_text(json.dumps(d), encoding="utf-8")

    names = class_names()
    base = {"path": str(out.resolve()), "names": names}
    _write_yaml(out / "sc_terran.yaml", {**base, "train": "train.txt", "val": "images/val",
                                          "test": "images/test"})
    for s in EVAL_SPLITS:
        if report["frames"][s]:
            _write_yaml(out / f"sc_terran_{s}.yaml",
                        {**base, "train": "train.txt", "val": f"images/{s}", "test": f"images/{s}"})

    result = {
        "frames": dict(report["frames"]),
        "skipped": dict(report["skipped"]),
        "train_images_after_repeat": sum(reps.values()),
        "classes": {s: {names[c]: n for c, n in sorted(cnt.items())}
                    for s, cnt in report["classes"].items()},
        "rare_classes_train": {names[c]: train_counts.get(c, 0) for c in names
                               if train_counts.get(c, 0) < cfg.min_instances_per_class},
    }
    (out / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def _write_yaml(path: Path, d: dict):
    path.write_text(yaml.safe_dump(d, allow_unicode=True, sort_keys=False), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, type=Path)
    ap.add_argument("--ann", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--gold", type=Path, default=None, help="사람이 고친 정밀 검수 주석 폴더 (평가 세트 ⑤)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--max-rejected-per-frame", type=int, default=0)
    ap.add_argument("--copy", action="store_true", help="하드 링크 대신 복사")
    args = ap.parse_args(argv)
    res = build(args.raw, args.ann, args.out, load_config(args.config), args.gold,
                args.max_rejected_per_frame, args.copy)
    print(json.dumps({k: res[k] for k in ("frames", "skipped", "train_images_after_repeat")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
