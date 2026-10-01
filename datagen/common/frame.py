"""라벨 수집 모듈이 기록하는 화면 1장 단위 JSON(프레임 기록)의 형식.

파일 쌍: {replay_id}_{frame:06d}.png  +  {replay_id}_{frame:06d}.json
형식 버전 1. 라벨 수집 모듈(tools/label_collector)과 반드시 같이 바꿀 것.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

FORMAT_VERSION = 1


@dataclass
class UnitRecord:
    id: int
    type: str                       # 게임 정보 인터페이스 라이브러리 유닛 종류 이름 (예: Terran_Marine)
    owner: int                      # 플레이어 번호
    color: int                      # 소유자 플레이어 색상 번호(팔레트 번호)
    x: int                          # 맵 좌표(픽셀), 유닛 중심
    y: int
    box: tuple[int, int, int, int]  # 충돌 사각형 (왼, 위, 오른, 아래), 맵 좌표
    hp: int = 0
    selected: bool = False
    flying: bool = False
    lifted: bool = False
    constructing: bool = False
    cloaked: bool = False
    burrowed: bool = False
    # 1프레임 전, 2프레임 전 맵 좌표 — 화면-상태 어긋남 실험용 (datagen/sam_masks/lag_check.py)
    prev: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class FrameRecord:
    replay_id: str
    frame: int
    camera: tuple[int, int]         # 화면 왼쪽 위의 맵 좌표
    units: list[UnitRecord]
    map_name: str = ""
    source: str = "replay"          # "replay" 또는 "live"(실제 대전 캡처, 평가 세트 ④)
    screen: tuple[int, int] = (640, 480)
    capture_lag_frames: int = 0
    vision_players: list[int] = field(default_factory=list)
    resources: list | None = None   # 2단계 숫자 인식용 정답 (플레이어별 광물, 가스, 보급)
    # 시야 밖이라 기록하지 않았지만 화면에 "마지막으로 본 모습"이 남아 있을 수 있는 건물 수.
    # 0보다 크면 라벨이 빠진 물체가 화면에 있을 수 있으므로 학습에서 뺀다.
    fogged_buildings: int = 0

    @property
    def stem(self) -> str:
        return f"{self.replay_id}_{self.frame:06d}"


def load_frame(path: str | Path) -> FrameRecord:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if d.get("version") != FORMAT_VERSION:
        raise ValueError(f"{path}: 지원하지 않는 형식 버전 {d.get('version')}")
    units = [
        UnitRecord(
            id=u["id"], type=u["type"], owner=u["owner"], color=u["color"],
            x=u["x"], y=u["y"], box=tuple(u["box"]), hp=u.get("hp", 0),
            selected=u.get("selected", False), flying=u.get("flying", False),
            lifted=u.get("lifted", False), constructing=u.get("constructing", False),
            cloaked=u.get("cloaked", False), burrowed=u.get("burrowed", False),
            prev=[tuple(p) for p in u.get("prev", [])],
        )
        for u in d["units"]
    ]
    return FrameRecord(
        replay_id=d["replay_id"], frame=d["frame"], camera=tuple(d["camera"]), units=units,
        map_name=d.get("map", ""), source=d.get("source", "replay"),
        screen=tuple(d.get("screen", (640, 480))),
        capture_lag_frames=d.get("capture_lag_frames", 0),
        vision_players=d.get("vision_players", []), resources=d.get("resources"),
        fogged_buildings=d.get("fogged_buildings", 0),
    )


def iter_frames(raw_dir: str | Path):
    """원본 폴더에서 (PNG 경로, 프레임 기록) 쌍을 순서대로 돌려준다. PNG가 없는 기록은 건너뛴다."""
    for jp in sorted(Path(raw_dir).rglob("*.json")):
        png = jp.with_suffix(".png")
        if png.exists():
            yield png, load_frame(jp)
