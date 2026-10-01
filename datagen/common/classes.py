"""테란 1차 41개 클래스 정의와 게임 정보 인터페이스 라이브러리(BWAPI) 유닛 이름 매핑."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Category(str, Enum):
    GROUND = "ground"      # 지상 유닛
    AIR = "air"            # 공중 유닛 (그림자가 생김)
    BUILDING = "building"  # 땅에 붙은 건물
    LIFTED = "lifted"      # 공중에 뜬 건물 (그림자가 생김)
    RESOURCE = "resource"  # 광물, 가스


@dataclass(frozen=True)
class ClassInfo:
    id: int
    name: str
    category: Category


_UNITS_GROUND = [
    "Terran_SCV", "Terran_Marine", "Terran_Firebat", "Terran_Medic", "Terran_Ghost",
    "Terran_Vulture", "Terran_Vulture_Spider_Mine", "Terran_Siege_Tank_Tank_Mode",
    "Terran_Siege_Tank_Siege_Mode", "Terran_Goliath",
]
_UNITS_AIR = [
    "Terran_Wraith", "Terran_Dropship", "Terran_Science_Vessel", "Terran_Battlecruiser",
    "Terran_Valkyrie",
]
_BUILDINGS = [
    "Terran_Command_Center", "Terran_Comsat_Station", "Terran_Nuclear_Silo",
    "Terran_Supply_Depot", "Terran_Refinery", "Terran_Barracks", "Terran_Engineering_Bay",
    "Terran_Missile_Turret", "Terran_Academy", "Terran_Bunker", "Terran_Factory",
    "Terran_Machine_Shop", "Terran_Starport", "Terran_Control_Tower",
    "Terran_Science_Facility", "Terran_Covert_Ops", "Terran_Physics_Lab", "Terran_Armory",
]
# 공중에 뜰 수 있는 건물. 뜬 상태는 모양이 완전히 달라 별도 클래스로 둔다.
LIFTABLE = [
    "Terran_Command_Center", "Terran_Barracks", "Terran_Factory", "Terran_Starport",
    "Terran_Science_Facility", "Terran_Engineering_Bay",
]
_RESOURCES = ["Mineral_Field", "Vespene_Geyser"]

# 광물은 모양이 다른 세 가지 유닛 종류가 있지만 하나의 클래스로 묶는다.
_RESOURCE_ALIASES = {
    "Resource_Mineral_Field": "Mineral_Field",
    "Resource_Mineral_Field_Type_2": "Mineral_Field",
    "Resource_Mineral_Field_Type_3": "Mineral_Field",
    "Resource_Vespene_Geyser": "Vespene_Geyser",
}

# 화면에 그려지는 물체가 아닌 유닛 종류 (가림 판단에서도 제외)
NON_VISUAL_PREFIXES = ("Spell_", "Special_Map_Revealer")


def _build() -> list[ClassInfo]:
    out: list[ClassInfo] = []
    for n in _UNITS_GROUND:
        out.append(ClassInfo(len(out), n, Category.GROUND))
    for n in _UNITS_AIR:
        out.append(ClassInfo(len(out), n, Category.AIR))
    for n in _BUILDINGS:
        out.append(ClassInfo(len(out), n, Category.BUILDING))
    for n in LIFTABLE:
        out.append(ClassInfo(len(out), n + "_Lifted", Category.LIFTED))
    for n in _RESOURCES:
        out.append(ClassInfo(len(out), n, Category.RESOURCE))
    return out


CLASSES: list[ClassInfo] = _build()
NAME_TO_CLASS: dict[str, ClassInfo] = {c.name: c for c in CLASSES}
assert len(CLASSES) == 41


def is_visual(unit_type: str) -> bool:
    return not unit_type.startswith(NON_VISUAL_PREFIXES)


def class_for_unit(unit_type: str, lifted: bool = False) -> ClassInfo | None:
    """유닛 종류 이름(BWAPI UnitType 이름)과 떠 있는지 여부로 클래스를 찾는다. 대상 밖이면 None."""
    name = _RESOURCE_ALIASES.get(unit_type, unit_type)
    if lifted and name in LIFTABLE:
        name += "_Lifted"
    return NAME_TO_CLASS.get(name)


def class_names() -> dict[int, str]:
    return {c.id: c.name for c in CLASSES}
