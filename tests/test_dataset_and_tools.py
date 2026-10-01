import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from datagen.build_dataset.build import build, repeat_factors
from datagen.build_dataset.split import assign_split
from datagen.common.config import DatagenConfig, MaskConfig, SplitConfig
from datagen.replay_filter.filter_replays import _parse, accept
from datagen.review.overlay import render
from datagen.sam_masks.pipeline import process_frame
from perception.detect.owner import TeamPalette, calibrate, polygon_mask
from perception.detect.train import resolve_config
from tests.synth import BlobSegmenter, draw, frame, unit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "capture"))
import sync_block as sb  # noqa: E402


def test_split_is_by_replay_and_deterministic():
    cfg = SplitConfig(held_out_maps=["Polypoid"])
    s = [assign_split(f"r{i:08x}", "Fighting Spirit", "replay", cfg) for i in range(2000)]
    assert s == [assign_split(f"r{i:08x}", "Fighting Spirit", "replay", cfg) for i in range(2000)]
    frac = {k: s.count(k) / len(s) for k in set(s)}
    assert 0.07 < frac["test"] < 0.13 and 0.07 < frac["val"] < 0.13
    assert assign_split("r1", "polypoid", "replay", cfg) == "unseen_maps"
    assert assign_split("r1", "Fighting Spirit", "live", cfg) == "live"


def test_repeat_factors():
    reps = repeat_factors({"a": {0}, "b": {0, 5}, "c": set()}, {0: 3000, 5: 300}, 3000)
    assert reps == {"a": 1, "b": 10, "c": 1}


def _make_raw_and_ann(tmp_path, n_replays=30):
    raw, ann_dir = tmp_path / "raw", tmp_path / "ann"
    raw.mkdir(); ann_dir.mkdir()
    for r in range(n_replays):
        fr = frame([unit(1, "Terran_SCV", 1100, 2100), unit(2, "Terran_Marine", 1200, 2150)],
                   replay_id=f"r{r:08x}")
        img = draw(fr)
        cv2.imwrite(str(raw / f"{fr.stem}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        ann = process_frame(img, fr, BlobSegmenter(), MaskConfig())
        (ann_dir / f"{fr.stem}.json").write_text(ann.to_json())
    return raw, ann_dir


def test_build_dataset(tmp_path):
    raw, ann_dir = _make_raw_and_ann(tmp_path)
    out = tmp_path / "ds"
    res = build(raw, ann_dir, out, DatagenConfig())
    assert sum(res["frames"].values()) == 30
    data = yaml.safe_load((out / "sc_terran.yaml").read_text())
    assert data["names"][1] == "Terran_Marine" and data["train"] == "train.txt"
    lbl = next((out / "labels" / "train").glob("*.txt")).read_text().splitlines()
    assert len(lbl) == 2
    cls, *coords = lbl[0].split()
    assert cls in {"0", "1"} and len(coords) % 2 == 0 and all(0 <= float(v) <= 1 for v in coords)
    # 객체 수가 목표(3000)보다 훨씬 적으므로 학습 화면은 반복 상한(20)까지 반복된다
    lines = (out / "train.txt").read_text().splitlines()
    assert len(lines) == 20 * res["frames"]["train"]
    attrs = json.loads((out / "attrs_train.json").read_text())
    assert next(iter(attrs.values()))[0]["color"] == 111


def test_review_render(tmp_path):
    raw, ann_dir = _make_raw_and_ann(tmp_path, 1)
    from datagen.sam_masks.pipeline import load_ann
    ann = load_ann(next(ann_dir.glob("*.json")))
    img = cv2.imread(str(next(raw.glob("*.png"))))
    out = render(img, ann)
    assert out.shape == (960, 2560, 3)


def test_owner_palette_calibrate_and_classify():
    rng = np.random.default_rng(0)
    red = [(200, 0, 0), (150, 0, 0)]
    blue = [(0, 0, 200), (0, 0, 150)]
    body = [(90, 90, 90), (120, 120, 120)]          # 두 팀 공통 몸통 색 → 팀 색상으로 뽑히면 안 됨

    def sprite(team):
        img = np.zeros((20, 20, 3), np.uint8)
        cols = body + (red if team == 0 else blue)
        idx = rng.integers(0, len(cols), (20, 20))
        for i, c in enumerate(cols):
            img[idx == i] = c
        return img

    samples = [(sprite(t), np.ones((20, 20), bool), 111 if t == 0 else 165)
               for t in [0, 1] * 20]
    pal = calibrate(samples, min_count=10)
    assert len(pal.colors[111]) == 2 and len(pal.colors[165]) == 2
    c, conf = pal.classify(sprite(1), np.ones((20, 20), bool))
    assert c == 165 and conf == 1.0
    assert pal.classify(np.zeros((5, 5, 3), np.uint8), np.ones((5, 5), bool)) == (None, 0.0)


def test_palette_save_load(tmp_path):
    pal = TeamPalette({111: {0xC80000}})
    pal.save(tmp_path / "p.json")
    assert TeamPalette.load(tmp_path / "p.json").colors == {111: {0xC80000}}
    assert polygon_mask([[0, 0], [4, 0], [4, 4], [0, 4]], (10, 10)).sum() == 25


def test_sync_block_roundtrip():
    buf = bytearray(sb.pack_config("D:/out", interval=12, seed=7))
    assert len(buf) == sb.SIZE == 680
    stem = b"r0000abcd_000120"
    buf[sb.OFF_REQUEST_FRAME: sb.OFF_REQUEST_FRAME + 4] = (120).to_bytes(4, "little")
    buf[sb.OFF_STEM: sb.OFF_STEM + len(stem)] = stem
    assert sb.read_request(buf) == (120, "r0000abcd_000120")
    sb.write_status(buf, sb.STATUS_OK)
    assert int.from_bytes(buf[sb.OFF_STATUS: sb.OFF_STATUS + 4], "little") == 1


def test_replay_filter_parse():
    d = {"Header": {"Map": "Fighting Spirit", "Frames": 24 * 60 * 12,
                    "Players": [{"Race": {"Name": "Terran"}, "Type": {"Name": "Human"}},
                                {"Race": {"Name": "Terran"}, "Type": {"Name": "Human"}}]},
         "Computed": {"PlayerDescs": [{"APM": 180}, {"APM": 150}]}}
    info = _parse(Path("a.rep"), d)
    assert info.races == ["Terran", "Terran"] and 11 < info.minutes < 13
    assert accept(info)
    d["Header"]["Players"][1]["Race"]["Name"] = "Zerg"
    assert not accept(_parse(Path("a.rep"), d))


def test_train_config_resolution():
    cfg = resolve_config(None, "m1280", ["epochs=5", "data=/tmp/x.yaml"])
    assert cfg["model"] == "yolo11m-seg.pt" and cfg["imgsz"] == 1280 and cfg["name"] == "m1280"
    assert cfg["epochs"] == 5 and cfg["data"] == "/tmp/x.yaml"
    assert cfg["hsv_h"] == 0.0 and cfg["fliplr"] == 0.0
