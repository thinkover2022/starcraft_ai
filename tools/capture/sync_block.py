"""라벨 수집 모듈과 공유하는 동기화 블록. tools/label_collector/SyncBlock.h 와 같은 배치."""

from __future__ import annotations

import struct

MAGIC = 0x424C4353  # "SCLB"
VERSION = 1
MAPPING_NAME = "Local\\SCLabelSync"
REQUEST_EVENT = "Local\\SCCaptureRequest"
DONE_EVENT = "Local\\SCCaptureDone"

STATUS_PENDING, STATUS_OK, STATUS_FAILED = 0, 1, 2

# magic, version, interval, settle_frames, capture_lag_frames, vision_random, seed, timeout_ms,
# request_frame, status, stem[128], out_dir[512]
FMT = "<IIiiiiIiii128s512s"
SIZE = struct.calcsize(FMT)
assert SIZE == 4 * 10 + 128 + 512

OFF_REQUEST_FRAME = 4 * 8
OFF_STATUS = 4 * 9
OFF_STEM = 4 * 10


def pack_config(out_dir: str, interval=12, settle_frames=2, capture_lag_frames=0,
                vision_random=False, seed=0, timeout_ms=5000) -> bytes:
    od = out_dir.encode("utf-8")
    if len(od) >= 512:
        raise ValueError("저장 폴더 경로가 너무 김")
    return struct.pack(FMT, MAGIC, VERSION, interval, settle_frames, capture_lag_frames,
                       int(vision_random), seed, timeout_ms, -1, STATUS_PENDING, b"", od)


def read_request(buf) -> tuple[int, str]:
    frame = struct.unpack_from("<i", buf, OFF_REQUEST_FRAME)[0]
    stem = bytes(buf[OFF_STEM: OFF_STEM + 128]).split(b"\0", 1)[0].decode("ascii")
    return frame, stem


def write_status(buf, status: int):
    struct.pack_into("<i", buf, OFF_STATUS, status)
