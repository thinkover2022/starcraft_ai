"""실행용 엔진 변환 (문서 1.7절). 학습·실행과 같은 윈도우 컴퓨터(RTX 3090)에서 돌린다.

    python -m perception.detect.export --weights best.pt --imgsz 640 --format engine
    python -m perception.detect.export --weights best.pt --format onnx      # 대안

엔비디아 추론 최적화 엔진(TensorRT) 파일은 변환한 그래픽 카드·드라이버에서만 쓸 수 있다.
변환 후 --bench 로 화면 1장 추론 시간(목표 12밀리초 이하)을 잰다.
"""

from __future__ import annotations

import argparse
import time

import numpy as np


def bench(engine_path: str, imgsz: int, n: int = 300) -> dict:
    from ultralytics import YOLO

    model = YOLO(engine_path, task="segment")
    img = np.random.randint(0, 255, (480, 640, 3), np.uint8)
    for _ in range(20):
        model.predict(img, imgsz=imgsz, verbose=False)
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        model.predict(img, imgsz=imgsz, verbose=False)
        ts.append((time.perf_counter() - t) * 1000)
    ts = np.array(ts)
    return {"mean_ms": float(ts.mean()), "p95_ms": float(np.percentile(ts, 95))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--format", choices=["engine", "onnx"], default="engine")
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args(argv)

    from ultralytics import YOLO

    path = YOLO(args.weights).export(format=args.format, half=args.format == "engine",
                                     imgsz=args.imgsz, dynamic=False)
    print(f"변환 완료: {path}")
    if args.bench:
        print(bench(path, args.imgsz))


if __name__ == "__main__":
    main()
