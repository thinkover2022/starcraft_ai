"""평가 세트 ②–⑤ 일괄 평가 (문서 1.6절, 1.1절 완료 기준).

    python -m perception.detect.evaluate --weights runs/terran/s640/weights/best.pt \
        --dataset D:/sc_data/sc_terran_v1 --imgsz 640 --out runs/terran/s640/eval.json

데이터셋 폴더의 sc_terran_{test,unseen_maps,live,gold}.yaml 중 있는 것을 모두 평가한다.
지표: 박스·마스크 평균 정밀도 평균(IoU 0.5, IoU 0.5–0.95), 클래스별 박스 IoU 0.5 평균 정밀도,
      화면 1장 추론 시간, 완료 기준 통과 여부.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

EVAL_SPLITS = ["test", "unseen_maps", "live", "gold"]

# 문서 1.1절 완료 기준
TARGETS = {"box_map50": 0.90, "mask_map50": 0.85, "latency_ms": 12.0}


def evaluate(weights: str, dataset: Path, imgsz: int, device=0, half=True) -> dict:
    from ultralytics import YOLO

    model = YOLO(weights)
    out = {}
    for split in EVAL_SPLITS:
        data = dataset / f"sc_terran_{split}.yaml"
        if not data.exists():
            continue
        m = model.val(data=str(data), split="test", imgsz=imgsz, device=device, half=half,
                      plots=True, name=f"eval_{split}")
        names = m.names
        per_class = {names[int(c)]: float(ap) for c, ap in zip(m.box.ap_class_index, m.box.ap50)}
        latency = float(sum(m.speed.values()))   # 전처리 + 추론 + 후처리, 밀리초/장
        out[split] = {
            "box_map50": float(m.box.map50), "box_map50_95": float(m.box.map),
            "mask_map50": float(m.seg.map50), "mask_map50_95": float(m.seg.map),
            "latency_ms": latency, "per_class_box_ap50": per_class,
            "pass": {
                "box_map50": float(m.box.map50) >= TARGETS["box_map50"],
                "mask_map50": float(m.seg.map50) >= TARGETS["mask_map50"],
            },
        }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=0)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    res = evaluate(args.weights, args.dataset, args.imgsz, args.device)
    text = json.dumps(res, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
