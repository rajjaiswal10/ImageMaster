from __future__ import annotations

import logging
import uuid
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
EDSR_MODEL_PATH = MODEL_DIR / "EDSR_x2.pb"
REAL_ESRGAN_MODEL_PATH = MODEL_DIR / "RealESRGAN_x2plus.pth"
REAL_ESRGAN_MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"

# Prefer local models that a user may have placed in the app's models folder.
EDSR_MODEL_CANDIDATES = (
    MODEL_DIR / "EDSR_x2.pb",
    MODEL_DIR / "edsr_x2.pb",
    MODEL_DIR / "ESPCN_x2.pb",
    MODEL_DIR / "espcn_x2.pb",
)
REAL_ESRGAN_MODEL_CANDIDATES = (
    MODEL_DIR / "RealESRGAN_x2plus.pth",
    MODEL_DIR / "realesrgan-x2plus.pth",
    MODEL_DIR / "RealESRGAN_x2plus.pt",
    MODEL_DIR / "realesrgan-x2plus.pt",
)
INFERENCE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _find_first_valid_model(candidates: tuple[Path, ...]) -> Path | None:
    for path in candidates:
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def _ensure_superres_model() -> Path | None:
    local = _find_first_valid_model(EDSR_MODEL_CANDIDATES)
    if local is not None:
        return local
    return None


def _ensure_real_esrgan_model() -> Path | None:
    local = _find_first_valid_model(REAL_ESRGAN_MODEL_CANDIDATES)
    if local is not None:
        return local

    try:
        if not REAL_ESRGAN_MODEL_PATH.exists() or REAL_ESRGAN_MODEL_PATH.stat().st_size == 0:
            urlretrieve(REAL_ESRGAN_MODEL_URL, REAL_ESRGAN_MODEL_PATH)
        if REAL_ESRGAN_MODEL_PATH.exists() and REAL_ESRGAN_MODEL_PATH.stat().st_size > 0:
            return REAL_ESRGAN_MODEL_PATH
    except Exception:
        try:
            REAL_ESRGAN_MODEL_PATH.unlink(missing_ok=True)
        except Exception:
            pass
    return None


def pixel_unshuffle(x: torch.Tensor, scale: int) -> torch.Tensor:
    b, c, h, w = x.shape
    pad_h = (scale - h % scale) % scale
    pad_w = (scale - w % scale) % scale
    if pad_h > 0 or pad_w > 0:
        x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        _, _, h, w = x.shape

    out_c = c * scale * scale
    out_h = h // scale
    out_w = w // scale
    x_view = x.reshape(b, c, out_h, scale, out_w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_c, out_h, out_w)


class ResidualDenseBlock(torch.nn.Module):
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = torch.nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = torch.nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = torch.nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), dim=1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), dim=1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), dim=1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), dim=1))
        return x5 * 0.2 + x


class RRDB(torch.nn.Module):
    def __init__(self, num_feat: int, num_grow_ch: int = 32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(torch.nn.Module):
    def __init__(self, num_in_ch: int = 3, num_out_ch: int = 3, scale: int = 2, num_feat: int = 64, num_block: int = 23, num_grow_ch: int = 32):
        super().__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = torch.nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = torch.nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = torch.nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


def _repair_alpha_edges(alpha: Image.Image, target_size: tuple[int, int] | None = None) -> Image.Image:
    if alpha.mode != "L":
        alpha = alpha.convert("L")
    if target_size is not None and (alpha.width, alpha.height) != target_size:
        alpha = alpha.resize(target_size, resample=Image.Resampling.BICUBIC)

    arr = np.asarray(alpha, dtype=np.uint8)
    if arr.size == 0:
        return alpha

    # Remove isolated pixels and tiny clusters that commonly appear when resizing
    # vector/line-art masks. Keep the outer silhouette intact and preserve the
    # antialiased edge ramp instead of turning it into a noisy, broken outline.
    median = cv2.medianBlur(arr, 3)
    binary = cv2.threshold(median, 12, 255, cv2.THRESH_BINARY)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels > 1:
        min_area = max(4, int(binary.size * 0.0003))
        for label_index in range(1, num_labels):
            if stats[label_index, cv2.CC_STAT_AREA] < min_area:
                binary[labels == label_index] = 0

    soft = cv2.GaussianBlur(median, (3, 3), 0)
    repaired = np.where(binary > 0, np.maximum(binary, soft), np.minimum(binary, soft))
    repaired = cv2.GaussianBlur(repaired.astype(np.float32), (3, 3), 0)
    repaired = np.clip(repaired, 0, 255).astype(np.uint8)
    return Image.fromarray(repaired, mode="L")


def _apply_real_esrgan_model(image_rgb: Image.Image) -> Image.Image | None:
    model_path = _ensure_real_esrgan_model()
    if model_path is None:
        logger.info("Real-ESRGAN weights not available; using fallback upscale path.")
        return None

    try:
        if image_rgb.mode != "RGB":
            if "A" in image_rgb.getbands():
                image_rgb = _prepare_model_input(image_rgb)
            else:
                image_rgb = image_rgb.convert("RGB")

        state = torch.load(str(model_path), map_location=INFERENCE_DEVICE)
        if isinstance(state, dict) and "params_ema" in state:
            state_dict = state["params_ema"]
        elif isinstance(state, dict) and "params" in state:
            state_dict = state["params"]
        else:
            state_dict = state

        if not isinstance(state_dict, dict):
            logger.info("Real-ESRGAN weight file format unsupported: %s", type(state_dict).__name__)
            return None

        cleaned = {}
        for key, value in state_dict.items():
            cleaned[key.replace("module.", "").replace("model.", "")] = value

        model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=2, num_feat=64, num_block=23, num_grow_ch=32)
        load_result = model.load_state_dict(cleaned, strict=False)
        if load_result.missing_keys or load_result.unexpected_keys:
            logger.warning(
                "Real-ESRGAN checkpoint is incompatible; missing=%s unexpected=%s",
                load_result.missing_keys,
                load_result.unexpected_keys,
            )
            return None
        model = model.to(INFERENCE_DEVICE)
        model.eval()

        arr = np.asarray(image_rgb, dtype=np.float32) / 255.0
        orig_h, orig_w = arr.shape[:2]
        pad_h = (2 - orig_h % 2) % 2
        pad_w = (2 - orig_w % 2) % 2
        if pad_h > 0 or pad_w > 0:
            arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')

        tensor = torch.from_numpy(np.ascontiguousarray(np.transpose(arr, (2, 0, 1)))).unsqueeze(0).to(INFERENCE_DEVICE)
        with torch.no_grad():
            output = model(tensor).clamp(0.0, 1.0)
        out = output[0].detach().cpu().permute(1, 2, 0).contiguous().numpy()
        out = out[:orig_h * 2, :orig_w * 2]
        if out.shape != (orig_h * 2, orig_w * 2, 3) or not np.isfinite(out).all():
            logger.warning("Real-ESRGAN returned an invalid output shape or pixel array.")
            return None
        out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(out, "RGB")
    except Exception:
        logger.exception("Real-ESRGAN inference failed.")
        return None


def _prepare_model_input(image_rgba: Image.Image) -> Image.Image:
    if "A" not in image_rgba.getbands():
        return image_rgba.convert("RGB")

    matte = Image.new("RGBA", image_rgba.size, (255, 255, 255, 255))
    matte = Image.alpha_composite(matte, image_rgba)
    return matte.convert("RGB")


def _has_real_transparency(rgba: Image.Image) -> bool:
    alpha = rgba.getchannel("A")
    extrema = alpha.getextrema()
    return extrema[0] < 255 or extrema[1] < 255


def _fallback_upscale(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    target_size = (rgba.width * 2, rgba.height * 2)

    has_transparency = _has_real_transparency(rgba)
    if not has_transparency:
        rgb = rgba.convert("RGB").resize(target_size, resample=Image.Resampling.LANCZOS)
        rgb = rgb.filter(ImageFilter.GaussianBlur(radius=0.2))
        result = rgb.convert("RGBA")
        result.putalpha(Image.new("L", target_size, 255))
        return result

    rgb_array = np.asarray(rgba.convert("RGB"), dtype=np.float32)
    alpha_array = np.asarray(rgba.getchannel("A"), dtype=np.float32) / 255.0
    premultiplied = rgb_array * alpha_array[..., None]
    resized_rgb = cv2.resize(premultiplied, target_size, interpolation=cv2.INTER_LANCZOS4)
    resized_alpha = cv2.resize(alpha_array, target_size, interpolation=cv2.INTER_LANCZOS4)
    resized_alpha = np.clip(resized_alpha, 0.0, 1.0)
    rgb_array = np.divide(
        resized_rgb,
        np.maximum(resized_alpha[..., None], 1 / 255),
        out=np.zeros_like(resized_rgb),
        where=resized_alpha[..., None] > 1 / 255,
    )
    rgb_array = np.clip(rgb_array, 0, 255).astype(np.uint8)
    rgb = Image.fromarray(rgb_array, "RGB")
    alpha = _repair_alpha_edges(Image.fromarray((resized_alpha * 255).astype(np.uint8), "L"), target_size)
    result = rgb.convert("RGBA")
    result.putalpha(alpha)
    return result


def upscale_image(source_path: str | Path, output_dir: str | Path, mode: str = "ai") -> Path:
    source = Path(source_path)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(source) as img:
        image_rgba = img.convert("RGBA")
        has_alpha_channel = "A" in img.getbands()
        has_transparency = has_alpha_channel and _has_real_transparency(image_rgba)

        if mode not in {"ai", "normal"}:
            raise ValueError("Upscale mode must be 'ai' or 'normal'")

        if mode == "normal":
            result = _fallback_upscale(image_rgba)
        else:
            model_input_rgb = _prepare_model_input(image_rgba) if has_transparency else image_rgba.convert("RGB")
            result = _apply_real_esrgan_model(model_input_rgb)
            if result is None:
                result = _fallback_upscale(image_rgba)

        if has_alpha_channel and mode == "ai":
            alpha = _repair_alpha_edges(image_rgba.getchannel("A"), result.size)
            model_rgb = np.asarray(result.convert("RGB"), dtype=np.float32)
            edge_safe_rgb = np.asarray(_fallback_upscale(image_rgba).convert("RGB"), dtype=np.float32)
            alpha_array = np.asarray(alpha, dtype=np.float32) / 255.0
            model_weight = np.clip((alpha_array - 0.82) / 0.16, 0.0, 1.0)[..., None]
            blended_rgb = edge_safe_rgb * (1.0 - model_weight) + model_rgb * model_weight
            result = Image.fromarray(np.clip(blended_rgb, 0, 255).astype(np.uint8), "RGB").convert("RGBA")
            result.putalpha(alpha)
        out_path = target_dir / f"{uuid.uuid4().hex}.png"
        result.save(out_path, format="PNG")
        return out_path
