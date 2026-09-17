# Object Cutter

A local Flask web app for:

1. Instant background removal with alpha matting.
2. Click-to-select object extraction.
3. Automatic extraction of multiple objects.
4. Multiple image and ZIP uploads.

Designed as a starting point for low-VRAM GPUs such as GTX 1650.

## Current model choices

### Background removal
`rembg` + `u2netp` by default.

`u2netp` is deliberately selected for lower VRAM. If quality is more important than speed, edit:

```python
# processing/background.py
_SESSION = new_session("u2net")
```

### Automatic multiple-object extraction
Ultralytics `yolov8n-seg.pt`.

The model downloads automatically on its first use.

### Click selection
The first version uses OpenCV GrabCut so the project has no mandatory large segmentation checkpoint.

For a later quality upgrade, replace `processing/click_segment.py` with MobileSAM/SAM-based prompting. The frontend/API already separates this functionality.

## Windows setup

Open PowerShell in this folder:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

For access from another device on your LAN:

```text
http://YOUR-PC-IP:5000
```

## GPU note

`ultralytics` will normally use CUDA if a CUDA-enabled PyTorch installation is available.

Check:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

If it says `False`, the app still works, but inference will run on CPU.

For a GTX 1650, start with the small models and process one image at a time.

## ZIP behavior

A ZIP is extracted into an isolated temporary job directory. Supported image files are:

PNG, JPG, JPEG, WEBP, BMP.

Nested directories are preserved.

## Security notes

This is intended for local/private use.

Before exposing it to the public internet, add:
- authentication
- rate limits
- stricter ZIP validation
- file size/count limits
- cleanup jobs for old uploads/outputs
- CSRF protection if browser sessions are used
- production WSGI server

## Recommended next upgrade

For significantly better click-selection quality, integrate MobileSAM or a small SAM checkpoint into `processing/click_segment.py`.

The intended final pipeline is:

Upload
  -> cached image embedding
  -> click
  -> precise mask
  -> edge refinement
  -> transparent PNG

The automatic mode can also become:

YOLO detector
  -> SAM refinement
  -> individual object PNGs
