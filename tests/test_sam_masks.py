import numpy as np

from datagen.common.config import MaskConfig
from datagen.sam_masks.lag_check import lag_errors
from datagen.sam_masks.overlap import resolve_overlap
from datagen.sam_masks.pipeline import load_ann, process_frame
from datagen.sam_masks.polygon import mask_bbox, mask_to_polygon, polygon_to_yolo
from datagen.sam_masks.quality import AREA_RATIO, LOW_SCORE, NO_CENTER, check_raw_mask
from tests.synth import BlobSegmenter, draw, frame, unit


def test_overlap_gives_pixels_to_top():
    a = np.zeros((10, 10), bool); a[0:6, 0:6] = True
    b = np.zeros((10, 10), bool); b[3:9, 3:9] = True
    out, frac = resolve_overlap(np.stack([a, b]), [(0, 5), (0, 8)])   # b가 위
    assert out[1].sum() == 36
    assert out[0].sum() == 36 - 9
    assert np.isclose(frac[0], 27 / 36) and np.isclose(frac[1], 1.0)
    assert not (out[0] & out[1]).any()


def test_polygon_and_bbox():
    m = np.zeros((20, 30), bool); m[5:15, 10:20] = True
    poly = mask_to_polygon(m)
    assert poly.shape[1] == 2 and len(poly) >= 4
    assert mask_bbox(m) == (10, 5, 20, 15)
    yolo = polygon_to_yolo(poly, 30, 20)
    assert all(0 <= v <= 1 for v in yolo) and len(yolo) == 2 * len(poly)
    assert mask_to_polygon(np.zeros((5, 5), bool)) is None


def test_polygon_merges_large_second_piece():
    m = np.zeros((20, 40), bool); m[5:15, 2:12] = True; m[5:15, 25:33] = True
    merged = mask_to_polygon(m, merge_ratio=0.3)
    single = mask_to_polygon(m, merge_ratio=1.1)
    assert merged[:, 0].max() > 30 and single[:, 0].max() < 15


def test_quality_flags():
    cfg = MaskConfig()
    m = np.zeros((50, 50), bool); m[10:20, 10:20] = True
    assert check_raw_mask(m, 0.9, (15, 15), 100, cfg) == []
    assert LOW_SCORE in check_raw_mask(m, 0.5, (15, 15), 100, cfg)
    assert NO_CENTER in check_raw_mask(m, 0.9, (40, 40), 100, cfg)
    assert AREA_RATIO in check_raw_mask(m, 0.9, (15, 15), 1000, cfg)


def _scene():
    units = [
        unit(1, "Terran_SCV", 1100, 2100),
        unit(2, "Terran_Marine", 1110, 2108),                     # SCV 아래쪽이라 위에 그려짐
        unit(3, "Terran_Wraith", 1300, 2150, half=(10, 10), flying=True),
        unit(4, "Terran_Ghost", 1400, 2200, cloaked=True),
        unit(5, "Zerg_Zergling", 1200, 2300),                      # 대상 클래스 아님 → 가림에만 사용
        unit(6, "Terran_Marine", 1203, 2297),                      # 저글링보다 위쪽이라 대부분 가려짐
        unit(7, "Terran_Command_Center", 1500, 2250, half=(30, 20)),
    ]
    return frame(units)


def test_process_frame_end_to_end():
    fr = _scene()
    img = draw(fr)
    ann = process_frame(img, fr, BlobSegmenter(), MaskConfig())
    by_id = {o.unit_id: o for o in ann.objects}
    assert set(by_id) == {1, 2, 3, 7}
    # 박스는 충돌 사각형이 아니라 실제로 그려진 픽셀(마스크)에서 다시 계산된다
    ys, xs = np.nonzero(np.all(img == img[108, 110], axis=-1))
    assert by_id[2].bbox == (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    assert len(by_id[1].polygon) > 4                   # 해병에 모서리가 가려져 사각형이 아님
    assert by_id[3].class_id == 10                     # 레이스
    reasons = {i["unit_id"]: i["reason"] for i in ann.ignored}
    assert reasons[4] == "cloaked"
    assert reasons[6] == "occluded"
    assert not ann.frame_rejected


def test_console_area_is_cut():
    fr = frame([unit(1, "Terran_Marine", 1100, 2395)])
    ann = process_frame(draw(fr), fr, BlobSegmenter(), MaskConfig(play_area_bottom=400))
    assert ann.objects[0].bbox[3] == 400


def test_fogged_frame_rejected():
    fr = _scene()
    fr.fogged_buildings = 1
    ann = process_frame(draw(fr), fr, BlobSegmenter(), MaskConfig())
    assert ann.frame_rejected and ann.reject_reason == "fogged_buildings"


def test_annotation_roundtrip(tmp_path):
    fr = _scene()
    ann = process_frame(draw(fr), fr, BlobSegmenter(), MaskConfig())
    p = tmp_path / "a.json"
    p.write_text(ann.to_json())
    back = load_ann(p)
    assert back.stem == ann.stem and len(back.objects) == len(ann.objects)
    assert tuple(back.objects[0].bbox) == tuple(ann.objects[0].bbox)


def test_lag_check_finds_one_frame_lag():
    # 화면에는 1프레임 전 위치가 그려져 있는 상황을 만든다
    units = []
    for i in range(6):
        x, y = 1100 + i * 40, 2100
        u = unit(i + 1, "Terran_Marine", x, y, prev=[(x - 4, y), (x - 8, y)])
        units.append(u)
    fr = frame(units)
    shown = frame([unit(u.id, u.type, u.prev[0][0], u.prev[0][1]) for u in units])
    ann = process_frame(draw(shown), fr, BlobSegmenter(), MaskConfig())
    errs = lag_errors([(fr, ann)])
    med = {k: np.median(v) for k, v in errs.items()}
    assert min(med, key=med.get) == 1
