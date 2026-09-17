from pathlib import Path
import uuid
from PIL import Image
from ultralytics import YOLO
import cv2
import numpy as np
import torch

from .edges import feather_alpha

_MODEL = None

# Small segmentation model: suitable for lower-VRAM machines.
MODEL_NAME = "yolov8n-seg.pt"

def _model():
    global _MODEL
    if _MODEL is None:
        _MODEL = YOLO(MODEL_NAME)
    return _MODEL


def extract_all_objects(src: Path, output_dir: Path):
    model = _model()
    device = 0 if torch.cuda.is_available() else "cpu"
    results = model.predict(
        source=str(src),
        conf=0.25,
        iou=0.5,
        retina_masks=True,
        verbose=False,
        device=device,
    )

    result = results[0]
    image = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not read image")

    objects = []
    if result.masks is None:
        return objects

    names = result.names
    masks = result.masks.data.cpu().numpy()

    for idx, mask in enumerate(masks):
        # Resize mask to source resolution.
        mask = cv2.resize(
            mask.astype(np.uint8),
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        alpha = (mask > 0).astype(np.uint8) * 255

        # Anti-alias the mask boundary instead of a hard pixel cutoff.
        alpha = feather_alpha(alpha, feather=3)

        rgba = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        rgba[:, :, 3] = alpha

        cls = int(result.boxes.cls[idx].item())
        conf = float(result.boxes.conf[idx].item())
        label = names.get(cls, str(cls))

        ys, xs = np.where(alpha > 10)
        if len(xs) == 0:
            continue

        x1, y1, x2, y2 = map(int, result.boxes.xyxy[idx].cpu().numpy())
        crop = rgba[max(0,y1):min(rgba.shape[0],y2+1),
                    max(0,x1):min(rgba.shape[1],x2+1)]

        filename = f"{uuid.uuid4().hex}_{label}.png"
        out = output_dir / filename
        Image.fromarray(crop).save(out, "PNG")

        objects.append({
            "id": idx + 1,
            "label": label,
            "confidence": round(conf, 3),
            "bbox": [x1, y1, x2, y2],
            "url": f"/api/output/{filename}",
        })

    return objects
