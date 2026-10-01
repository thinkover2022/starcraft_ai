"""객체 인식 모델 학습 (문서 1.7절). 우분투 + RTX 3090, PyTorch + ultralytics.

    # 기본 실험 하나
    python -m perception.detect.train --config perception/detect/configs/train_s640.yaml
    # 실험 행렬 중 하나 (n640, s640, m640, n1280, s1280, m1280)
    python -m perception.detect.train --experiment s1280
    # 설정 값 덮어쓰기
    python -m perception.detect.train --set epochs=10 data=/data/pilot/sc_terran.yaml

학습이 끝나면 runs/terran/{name}/ 아래에 best.pt, 학습 곡선, 그리고
재현을 위한 run_meta.json(최종 설정, 코드 커밋 번호, 데이터셋 report.json 사본)이 남는다.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).parent / "configs"


def _parse_value(v: str):
    return yaml.safe_load(v)


def resolve_config(config: Path | None, experiment: str | None, overrides: list[str]) -> dict:
    if experiment:
        exp = yaml.safe_load((CONFIG_DIR / "experiments.yaml").read_text())
        cfg = yaml.safe_load((CONFIG_DIR / exp["base"]).read_text())
        if experiment not in exp["experiments"]:
            raise SystemExit(f"알 수 없는 실험: {experiment} (가능: {list(exp['experiments'])})")
        cfg.update(exp["experiments"][experiment])
        cfg["name"] = experiment
    else:
        cfg = yaml.safe_load(Path(config or CONFIG_DIR / "train_s640.yaml").read_text())
    for kv in overrides:
        k, _, v = kv.partition("=")
        cfg[k] = _parse_value(v)
    return cfg


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--experiment")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--dry-run", action="store_true", help="최종 설정만 출력")
    args = ap.parse_args(argv)

    cfg = resolve_config(args.config, args.experiment, args.set)
    if args.dry_run:
        print(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
        return

    from ultralytics import YOLO

    model_path = cfg.pop("model")
    model = YOLO(model_path)
    results = model.train(**cfg)

    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else Path(cfg["project"]) / cfg["name"]
    meta = {"model": model_path, "config": cfg, "git_commit": _git_commit()}
    report = Path(cfg["data"]).parent / "report.json"
    if report.exists():
        meta["dataset_report"] = json.loads(report.read_text())
    (save_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"학습 완료: {save_dir}")


if __name__ == "__main__":
    main()
