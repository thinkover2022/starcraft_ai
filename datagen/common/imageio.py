"""윈도우에서도 안전한 이미지 읽기·쓰기.

윈도우의 cv2.imread / cv2.imwrite 는 경로에 한글 등 비ASCII 문자가 있으면 오류 없이 실패한다
(None 반환 / False 반환). 그래서 파일 입출력은 numpy 로 하고 OpenCV 는 인코딩·디코딩만 한다.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """BGR(또는 flags 에 따른) 이미지. 읽지 못하면 예외."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    if img is None:
        raise IOError(f"이미지를 읽을 수 없음: {path}")
    return img


def imread_rgb(path: str | Path) -> np.ndarray:
    return cv2.cvtColor(imread(path), cv2.COLOR_BGR2RGB)


def imwrite(path: str | Path, img: np.ndarray) -> None:
    """확장자로 형식을 정한다 (.png 는 무손실). 쓰지 못하면 예외."""
    path = Path(path)
    ok, buf = cv2.imencode(path.suffix or ".png", img)
    if not ok:
        raise IOError(f"이미지를 인코딩할 수 없음: {path}")
    buf.tofile(str(path))


def index_pngs(root: str | Path) -> dict[str, Path]:
    """폴더 아래 모든 PNG 를 한 번만 훑어 {파일 이름(확장자 제외): 경로} 로 돌려준다."""
    return {p.stem: p for p in Path(root).rglob("*.png")}
