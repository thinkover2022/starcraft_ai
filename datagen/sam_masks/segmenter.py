"""범용 분할 모델(Segment Anything Model) 사각형 힌트 분할기.

`BoxSegmenter` 규약만 지키면 다른 모델로 바꿀 수 있다. 시험에서는 가짜 분할기를 쓴다.
"""

from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np


class BoxSegmenter(Protocol):
    def segment(
        self, image_rgb: np.ndarray, boxes: np.ndarray,
        neg_points: list[tuple[float, float] | None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """image_rgb (H,W,3) uint8, boxes (N,4) 화면 좌표.

        반환: masks (N,H,W) bool, scores (N,) float — 입력 화면과 같은 해상도.
        """
        ...


class Sam2BoxSegmenter:
    """메타 범용 분할 모델 2.1 래퍼. 작은 유닛 때문에 화면을 `upscale`배 확대해서 넣는다."""

    def __init__(self, model_id: str = "facebook/sam2.1-hiera-large", upscale: int = 2,
                 device: str = "cuda"):
        import torch
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self._torch = torch
        self.device = device
        self.upscale = upscale
        self.predictor = SAM2ImagePredictor.from_pretrained(model_id, device=device)

    def segment(self, image_rgb, boxes, neg_points=None):
        n = len(boxes)
        h, w = image_rgb.shape[:2]
        if n == 0:
            return np.zeros((0, h, w), bool), np.zeros((0,), np.float32)
        s = self.upscale
        big = cv2.resize(image_rgb, (w * s, h * s), interpolation=cv2.INTER_CUBIC)
        kwargs = {"box": np.asarray(boxes, np.float32) * s, "multimask_output": False}
        if neg_points and any(p is not None for p in neg_points):
            # 음성 점이 없는 물체는 라벨 -1(빈 자리 채움)로 둔다.
            coords = np.zeros((n, 1, 2), np.float32)
            labels = np.full((n, 1), -1, np.int32)
            for i, p in enumerate(neg_points):
                if p is not None:
                    coords[i, 0] = (p[0] * s, p[1] * s)
                    labels[i, 0] = 0
            kwargs.update(point_coords=coords, point_labels=labels)

        torch = self._torch
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            self.predictor.set_image(big)
            masks, scores, _ = self.predictor.predict(**kwargs)
        masks = np.asarray(masks, np.float32).reshape(n, h * s, w * s)
        scores = np.asarray(scores, np.float32).reshape(n)
        small = np.stack([cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA) for m in masks])
        return small > 0.5, scores
