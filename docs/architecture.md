# Architecture

Frontend:
- Drag/drop
- image list
- mode switcher
- canvas click coordinates
- result cards

Backend:
- Flask REST endpoints
- isolated upload directories
- background removal
- click segmentation
- automatic segmentation

Processing:
- Background: rembg/u2netp
- Click: OpenCV GrabCut
- Auto: YOLOv8n-seg

The code intentionally keeps these processors independent so the click processor
can later be replaced by MobileSAM/SAM without changing the UI.
