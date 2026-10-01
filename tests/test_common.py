import numpy as np

from datagen.common.classes import CLASSES, Category, class_for_unit, is_visual
from datagen.common.config import MaskConfig, load_config
from datagen.common.frame import load_frame
from datagen.common.geometry import draw_order_key, play_area_mask, prompt_box, shadow_point
from tests.synth import CPP_SAMPLE, frame, unit


def test_class_table():
    assert len(CLASSES) == 41
    assert [c.id for c in CLASSES] == list(range(41))
    assert class_for_unit("Terran_SCV").id == 0
    assert class_for_unit("Terran_Marine").id == 1
    assert class_for_unit("Terran_Barracks", lifted=True).name == "Terran_Barracks_Lifted"
    assert class_for_unit("Terran_Barracks", lifted=True).category is Category.LIFTED
    assert class_for_unit("Terran_Supply_Depot", lifted=True).name == "Terran_Supply_Depot"
    assert class_for_unit("Resource_Mineral_Field_Type_3").name == "Mineral_Field"
    assert class_for_unit("Zerg_Zergling") is None
    assert not is_visual("Spell_Scanner_Sweep")


def test_load_frame_matches_cpp_output(tmp_path):
    p = tmp_path / "r1a2b3c4d_002400.json"
    p.write_text(CPP_SAMPLE)
    f = load_frame(p)
    assert f.stem == "r1a2b3c4d_002400"
    assert f.camera == (1200, 900)
    assert f.units[0].box == (1491, 1090, 1508, 1109)
    assert f.units[0].prev == [(1498, 1100), (1496, 1100)]
    assert f.resources[0]["minerals"] == 350


def test_prompt_box_converts_and_clips():
    cfg = MaskConfig()
    fr = frame([unit(1, "Terran_Marine", 1010, 2010)])
    box = prompt_box(fr, fr.units[0], cfg)
    # 충돌 사각형 (1002..1017, 2002..2017) - 카메라 (1000, 2000) - 여유 폭 → 화면 경계에서 잘림
    assert box.tolist() == [0.0, 0.0, 21.0, 21.0]
    off = frame([unit(2, "Terran_Marine", 5000, 5000)])
    assert prompt_box(off, off.units[0], cfg) is None


def test_type_margin_override():
    cfg = MaskConfig(type_margins={"Terran_Marine": (0, 0, 0, 0)})
    fr = frame([unit(1, "Terran_Marine", 1100, 2100)])
    assert prompt_box(fr, fr.units[0], cfg).tolist() == [92.0, 92.0, 107.0, 107.0]


def test_draw_order_air_over_ground_and_lower_over_upper():
    g_hi = unit(1, "Terran_Marine", 100, 100)
    g_lo = unit(2, "Terran_Marine", 100, 120)
    air = unit(3, "Terran_Wraith", 100, 50, flying=True)
    mine = unit(4, "Terran_Vulture_Spider_Mine", 100, 200, burrowed=True)
    order = sorted([g_hi, g_lo, air, mine], key=draw_order_key)
    assert [u.id for u in order] == [4, 1, 2, 3]


def test_shadow_point_only_when_measured():
    fr = frame([unit(1, "Terran_Wraith", 1100, 2100, flying=True)])
    assert shadow_point(fr, fr.units[0], MaskConfig()) is None
    assert shadow_point(fr, fr.units[0], MaskConfig(shadow_offset=(0, 40))) == (100, 140)


def test_play_area_mask():
    assert play_area_mask((480, 640), MaskConfig()) is None
    m = play_area_mask((480, 640), MaskConfig(play_area_bottom=400))
    assert m[399].all() and not m[400].any()


def test_load_config(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("mask:\n  margins:\n    ground: [1, 2, 3, 4]\n  shadow_offset: [0, 42]\n"
                 "split:\n  held_out_maps: [Polypoid]\n")
    cfg = load_config(p)
    assert cfg.mask.margins["ground"] == (1, 2, 3, 4)
    assert cfg.mask.margins["building"] == (8, 32, 8, 8)    # 나머지는 기본값 유지
    assert cfg.mask.shadow_offset == (0, 42)
    assert cfg.split.held_out_maps == ["Polypoid"]


def test_repo_config_loads():
    cfg = load_config("configs/datagen.yaml")
    assert cfg.mask.type_margins["Terran_Siege_Tank_Siege_Mode"] == (16, 12, 16, 8)
    assert np.isclose(cfg.mask.min_score, 0.7)
