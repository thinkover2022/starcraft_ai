"""리플레이 걸러내기 (문서 1.5절): 테란 대 테란, 5분 이상, 분당 행동 수 80 이상.

    python -m datagen.replay_filter.filter_replays --replays D:/replays --out data/tvt_list.txt

리플레이 분석기 screp(https://github.com/icza/screp)이 PATH에 있어야 한다.
screp의 JSON 출력 형식은 버전에 따라 다를 수 있으므로, 처음 쓸 때
`screp 예시.rep` 출력과 아래 _parse()가 읽는 필드를 대조할 것.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

FPS_FASTEST = 1000 / 42  # 가장 빠름 속도의 게임 프레임/초


@dataclass
class ReplayInfo:
    path: str
    map_name: str
    minutes: float
    races: list[str]
    apms: list[int]


def _parse(path: Path, d: dict) -> ReplayInfo:
    h = d.get("Header", {})
    players = [p for p in h.get("Players", []) if (p.get("Type") or {}).get("Name") != "Observer"]
    races = [(p.get("Race") or {}).get("Name", "") for p in players]
    apms = [int(pd.get("APM", 0)) for pd in (d.get("Computed") or {}).get("PlayerDescs", [])]
    return ReplayInfo(str(path), h.get("Map", ""), h.get("Frames", 0) / FPS_FASTEST / 60, races, apms)


def read_replay(path: Path, screp: str = "screp") -> ReplayInfo | None:
    r = subprocess.run([screp, str(path)], capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        return None
    try:
        return _parse(path, json.loads(r.stdout))
    except (json.JSONDecodeError, TypeError):
        return None


def accept(info: ReplayInfo, races=("Terran", "Terran"), min_minutes=5.0, min_apm=80) -> bool:
    return (sorted(info.races) == sorted(races) and info.minutes >= min_minutes
            and len(info.apms) >= 2 and min(info.apms) >= min_apm)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replays", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--screp", default="screp")
    ap.add_argument("--min-minutes", type=float, default=5.0)
    ap.add_argument("--min-apm", type=int, default=80)
    args = ap.parse_args(argv)
    kept, maps = [], {}
    total = 0
    for p in sorted(args.replays.rglob("*.rep")):
        total += 1
        info = read_replay(p, args.screp)
        if info and accept(info, min_minutes=args.min_minutes, min_apm=args.min_apm):
            kept.append(info)
            maps[info.map_name] = maps.get(info.map_name, 0) + 1
    args.out.write_text("\n".join(i.path for i in kept) + "\n", encoding="utf-8")
    print(f"{total}개 중 {len(kept)}개 통과, 맵 {len(maps)}종")
    for m, n in sorted(maps.items(), key=lambda x: -x[1]):
        print(f"  {n:5d}  {m}")


if __name__ == "__main__":
    main()
