"""화면 캡처 프로그램 (문서 1.4절 [나]). 윈도우 전용.

라벨 수집 모듈보다 먼저 실행한다. 모듈이 "찍어라" 신호를 보내면 게임 창의 클라이언트 영역
(640x480)을 무손실 PNG로 저장하고 "다 찍었다" 신호를 돌려준다. 그동안 게임은 멈춰 있다.

    python tools/capture/capture_server.py --out D:/sc_raw/batch01 --interval 12 --settle 2

준비 사항
  - 스타크래프트 1.16.1을 창 모드 640x480, 확대 없이 실행 (예: 창 모드 플러그인 또는 DirectDraw 대체 래퍼)
  - 창이 다른 창에 가려지지 않게 둘 것 (dxcam 방식은 모니터 화면을 그대로 복사함)
  - 데이터 생성 중에는 마우스 커서를 게임 창 밖에 둘 것
"""

from __future__ import annotations

import argparse
import ctypes
import mmap
import sys
import time
from ctypes import wintypes
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 저장소 루트 (datagen 패키지)
import sync_block as sb  # noqa: E402

from datagen.common.imageio import imwrite  # noqa: E402  한글 경로에서도 저장되는 함수

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)

# 64비트 Python에서 핸들이 잘리지 않도록 반환·인자 형식을 명시한다.
_H = wintypes.HANDLE
kernel32.CreateEventW.restype = _H
kernel32.CreateEventW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]
kernel32.WaitForSingleObject.argtypes = [_H, wintypes.DWORD]
kernel32.WaitForSingleObject.restype = wintypes.DWORD
kernel32.SetEvent.argtypes = [_H]
user32.FindWindowW.restype = wintypes.HWND
user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
user32.GetDC.restype = wintypes.HDC
user32.GetDC.argtypes = [wintypes.HWND]
user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
gdi32.CreateCompatibleDC.restype = wintypes.HDC
gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
gdi32.CreateCompatibleBitmap.restype = wintypes.HBITMAP
gdi32.CreateCompatibleBitmap.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int]
gdi32.SelectObject.restype = wintypes.HGDIOBJ
gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
gdi32.BitBlt.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                         wintypes.HDC, ctypes.c_int, ctypes.c_int, wintypes.DWORD]
gdi32.GetDIBits.argtypes = [wintypes.HDC, wintypes.HBITMAP, wintypes.UINT, wintypes.UINT,
                            ctypes.c_void_p, ctypes.c_void_p, wintypes.UINT]
gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
gdi32.DeleteDC.argtypes = [wintypes.HDC]

WAIT_OBJECT_0 = 0
DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = ctypes.c_void_p(-4)


def set_dpi_aware():
    try:
        user32.SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)
    except AttributeError:
        pass


def find_window(cls: str | None, title: str | None) -> int:
    hwnd = user32.FindWindowW(cls, title)
    if not hwnd:
        raise SystemExit(f"게임 창을 찾을 수 없음 (class={cls!r}, title={title!r})")
    return hwnd


def client_rect_on_screen(hwnd) -> tuple[int, int, int, int]:
    r = wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(r))
    pt = wintypes.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(pt))
    return pt.x, pt.y, pt.x + r.right, pt.y + r.bottom


class DxcamGrabber:
    """직접 X 그래픽 인프라 화면 복제(DXGI Desktop Duplication) 방식."""

    def __init__(self, hwnd):
        import dxcam
        self.hwnd = hwnd
        self.cam = dxcam.create(output_color="BGR")
        self.last = None

    def grab(self) -> np.ndarray:
        region = client_rect_on_screen(self.hwnd)
        for _ in range(10):
            img = self.cam.grab(region=region)
            if img is not None:
                self.last = img
                return img
            time.sleep(0.002)
        # 새 화면이 없다 = 지난번 캡처 이후 화면이 변하지 않았다
        if self.last is None:
            raise RuntimeError("첫 화면을 얻지 못함")
        return self.last


class GdiGrabber:
    """GDI 비트 블록 복사 방식 (창 단위). 화면 복제 방식이 안 될 때의 예비."""

    def __init__(self, hwnd):
        self.hwnd = hwnd

    def grab(self) -> np.ndarray:
        r = wintypes.RECT()
        user32.GetClientRect(self.hwnd, ctypes.byref(r))
        w, h = r.right, r.bottom
        hdc = user32.GetDC(self.hwnd)
        mdc = gdi32.CreateCompatibleDC(hdc)
        bmp = gdi32.CreateCompatibleBitmap(hdc, w, h)
        gdi32.SelectObject(mdc, bmp)
        gdi32.BitBlt(mdc, 0, 0, w, h, hdc, 0, 0, 0x00CC0020)  # SRCCOPY

        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG),
                        ("biHeight", wintypes.LONG), ("biPlanes", wintypes.WORD),
                        ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                        ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
                        ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD),
                        ("biClrImportant", wintypes.DWORD)]

        bih = BITMAPINFOHEADER(ctypes.sizeof(BITMAPINFOHEADER), w, -h, 1, 32, 0, 0, 0, 0, 0, 0)
        buf = (ctypes.c_ubyte * (w * h * 4))()
        gdi32.GetDIBits(mdc, bmp, 0, h, buf, ctypes.byref(bih), 0)
        gdi32.DeleteObject(bmp)
        gdi32.DeleteDC(mdc)
        user32.ReleaseDC(self.hwnd, hdc)
        return np.frombuffer(buf, np.uint8).reshape(h, w, 4)[:, :, :3].copy()


def create_event(name: str):
    h = kernel32.CreateEventW(None, False, False, name)  # 자동 재설정 이벤트
    if not h:
        raise OSError(ctypes.get_last_error(), f"이벤트 생성 실패: {name}")
    return h


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--interval", type=int, default=12)
    ap.add_argument("--settle", type=int, default=2)
    ap.add_argument("--lag", type=int, default=0, help="실험으로 정한 화면-상태 어긋남 프레임 (기록용)")
    ap.add_argument("--vision-random", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout-ms", type=int, default=5000)
    ap.add_argument("--backend", choices=["dxcam", "gdi"], default="dxcam")
    ap.add_argument("--window-class", default="SWarClass")
    ap.add_argument("--window-title", default=None)
    ap.add_argument("--size", default="640x480", help="기대하는 클라이언트 영역 크기")
    args = ap.parse_args(argv)

    set_dpi_aware()
    args.out.mkdir(parents=True, exist_ok=True)
    exp_w, exp_h = map(int, args.size.split("x"))

    shm = mmap.mmap(-1, sb.SIZE, tagname=sb.MAPPING_NAME)
    shm[:] = sb.pack_config(str(args.out.resolve()), args.interval, args.settle, args.lag,
                            args.vision_random, args.seed, args.timeout_ms)
    req = create_event(sb.REQUEST_EVENT)
    done = create_event(sb.DONE_EVENT)
    print(f"대기 중… 이제 게임과 라벨 수집 모듈을 실행하세요. 저장 폴더: {args.out}")

    grabber = None
    n = 0
    while True:
        if kernel32.WaitForSingleObject(req, 1000) != WAIT_OBJECT_0:
            continue
        frame, stem = sb.read_request(shm)
        status = sb.STATUS_FAILED
        try:
            if grabber is None:
                hwnd = find_window(args.window_class, args.window_title)
                grabber = (DxcamGrabber if args.backend == "dxcam" else GdiGrabber)(hwnd)
            img = grabber.grab()
            if img.shape[:2] != (exp_h, exp_w):
                print(f"[경고] 캡처 크기 {img.shape[1]}x{img.shape[0]} ≠ {args.size} — 창 모드/확대 설정 확인")
            else:
                imwrite(args.out / f"{stem}.png", img)
                status = sb.STATUS_OK
                n += 1
        except Exception as e:  # 실패해도 서버는 계속 돈다. 모듈은 이 화면을 건너뛴다.
            print(f"[오류] {stem}: {e}")
            grabber = None
        sb.write_status(shm, status)
        kernel32.SetEvent(done)
        if n and n % 500 == 0:
            print(f"{n}장 저장 (마지막 프레임 {frame})", flush=True)


if __name__ == "__main__":
    main()
