from pathlib import Path
import uuid
import zipfile
import shutil
import json
import time
import re
import cv2
import numpy as np

from flask import Flask, render_template, request, jsonify, send_file, abort
from PIL import Image
from werkzeug.utils import secure_filename

from processing.background import remove_background
from processing.click_segment import segment_from_points, export_click_objects
from processing.auto_extract import extract_all_objects
from processing.upscale import upscale_image

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

ALLOWED_IMAGES = {"png", "jpg", "jpeg", "webp", "bmp"}
ALLOWED_ANIMATION_FORMATS = {"gif", "mp4"}

HISTORY_PATH = OUTPUT_DIR / "history.json"
MAX_HISTORY = 10

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 250 * 1024 * 1024


def allowed_image(name):
    return "." in name and name.rsplit(".", 1)[1].lower() in ALLOWED_IMAGES


def load_history():
    if not HISTORY_PATH.exists():
        return []
    try:
        return json.loads(HISTORY_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def save_history(entries, preserve_files=None):
    HISTORY_PATH.write_text(json.dumps(entries, indent=2))
    cleanup_output_dir(preserve_files)


def cleanup_output_dir(preserve_files=None):
    tracked = {e.get("filename") for e in load_history() if e.get("filename")}
    tracked.update(secure_filename(str(name)) for name in (preserve_files or []) if name)
    if not OUTPUT_DIR.exists():
        return
    for item in OUTPUT_DIR.iterdir():
        if item.name == HISTORY_PATH.name:
            continue
        if item.is_file() and item.name not in tracked:
            item.unlink(missing_ok=True)


def delete_output_file(filename):
    if not filename:
        return False
    target = (OUTPUT_DIR / secure_filename(filename)).resolve()
    if str(target).startswith(str(OUTPUT_DIR.resolve())) and target.is_file():
        target.unlink(missing_ok=True)
        return True
    return False


def delete_upload_source(image_id, job_id=None):
    if job_id:
        target = (UPLOAD_DIR / job_id / image_id).resolve()
        base = (UPLOAD_DIR / job_id).resolve()
        if str(target).startswith(str(base)) and target.is_file():
            target.unlink(missing_ok=True)
            try:
                parent = target.parent
                while parent != base:
                    if any(parent.iterdir()):
                        break
                    parent.rmdir()
                    parent = parent.parent
                if base.exists() and not any(base.iterdir()):
                    base.rmdir()
            except OSError:
                pass
            return True

    matches = list(UPLOAD_DIR.glob(f"{image_id}.*"))
    deleted = False
    for match in matches:
        if match.is_file():
            match.unlink(missing_ok=True)
            deleted = True
    return deleted


def add_history_entry(filename, mode, source_name):
    return add_history_entries([(filename, mode, source_name)])[0]


def add_history_entries(items, preserve_files=None):
    # Batch all new entries into a single save so cleanup_output_dir() doesn't
    # delete sibling files (e.g. other objects from the same auto-extract run)
    # that haven't been recorded in history yet.
    entries = load_history()
    added = []
    for filename, mode, source_name in items:
        entry = {
            "id": uuid.uuid4().hex,
            "filename": filename,
            "url": f"/api/output/{filename}",
            "mode": mode,
            "source_name": source_name,
            "created_at": time.time(),
        }
        entries.append(entry)
        added.append(entry)
    entries.sort(key=lambda e: e["created_at"])
    while len(entries) > MAX_HISTORY:
        removed = entries.pop(0)
        old_file = OUTPUT_DIR / removed["filename"]
        if old_file.is_file():
            old_file.unlink(missing_ok=True)
    save_history(entries, preserve_files=preserve_files)
    return added


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/upload")
def upload():
    files = request.files.getlist("files")
    if not files:
        return jsonify(error="No files supplied"), 400

    results = []
    for f in files:
        if not f.filename:
            continue

        suffix = Path(secure_filename(f.filename)).suffix.lower()
        if suffix == ".zip":
            job_id = uuid.uuid4().hex
            job_dir = UPLOAD_DIR / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            zip_path = job_dir / "input.zip"
            f.save(zip_path)

            with zipfile.ZipFile(zip_path) as z:
                for member in z.infolist():
                    # Prevent zip-slip and ignore directories.
                    target = (job_dir / member.filename).resolve()
                    if not str(target).startswith(str(job_dir.resolve())):
                        continue
                    if member.is_dir():
                        continue
                    if allowed_image(member.filename):
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with z.open(member) as src, open(target, "wb") as dst:
                            shutil.copyfileobj(src, dst)

            for image in job_dir.rglob("*"):
                if image.is_file() and allowed_image(image.name):
                    results.append({
                        "image_id": image.relative_to(job_dir).as_posix(),
                        "job_id": job_id,
                        "name": image.name,
                        "url": f"/api/source/{job_id}/{image.relative_to(job_dir).as_posix()}"
                    })
        elif allowed_image(f.filename):
            image_id = uuid.uuid4().hex
            suffix = Path(secure_filename(f.filename)).suffix.lower()
            path = UPLOAD_DIR / f"{image_id}{suffix}"
            f.save(path)
            results.append({
                "image_id": image_id,
                "job_id": None,
                "name": secure_filename(f.filename),
                "url": f"/api/source/{image_id}"
            })

    return jsonify(images=results)


@app.get("/api/source/<image_id>/<path:relative>")
def source_from_zip(image_id, relative):
    path = (UPLOAD_DIR / image_id / relative).resolve()
    base = (UPLOAD_DIR / image_id).resolve()
    if not str(path).startswith(str(base)) or not path.is_file():
        abort(404)
    return send_file(path)


@app.get("/api/source/<image_id>")
def source(image_id):
    matches = list(UPLOAD_DIR.glob(f"{image_id}.*"))
    if not matches:
        abort(404)
    return send_file(matches[0])


@app.delete("/api/source/<job_id>/<path:relative>")
def delete_source_from_zip(job_id, relative):
    path = (UPLOAD_DIR / job_id / relative).resolve()
    base = (UPLOAD_DIR / job_id).resolve()
    if not str(path).startswith(str(base)) or not path.is_file():
        return jsonify(error="Not found"), 404
    path.unlink(missing_ok=True)
    return jsonify(ok=True)


@app.delete("/api/source/<image_id>")
def delete_source(image_id):
    matches = list(UPLOAD_DIR.glob(f"{image_id}.*"))
    if not matches:
        return jsonify(error="Not found"), 404
    for m in matches:
        m.unlink(missing_ok=True)
    return jsonify(ok=True)


def resolve_image(image_id, job_id=None):
    if job_id:
        p = (UPLOAD_DIR / job_id / image_id).resolve()
        base = (UPLOAD_DIR / job_id).resolve()
        if str(p).startswith(str(base)) and p.is_file():
            return p
    matches = list(UPLOAD_DIR.glob(f"{image_id}.*"))
    return matches[0] if matches else None


def resolve_animation_source(source):
    if not isinstance(source, dict):
        return None
    if source.get("file_name"):
        name = secure_filename(str(source["file_name"]))
        path = (OUTPUT_DIR / name).resolve()
        if str(path).startswith(str(OUTPUT_DIR.resolve())) and path.is_file():
            return path
    return resolve_image(source.get("image_id"), source.get("job_id"))


def build_animation_frames(sources):
    loaded = []
    for source in sources:
        path = resolve_animation_source(source)
        if path is None:
            continue
        with Image.open(path) as image:
            frame = image.convert("RGBA")
            loaded.append(frame.copy())
    if not loaded:
        raise ValueError("No valid images supplied")

    width = max(frame.width for frame in loaded)
    height = max(frame.height for frame in loaded)
    frames = []
    for frame in loaded:
        canvas = Image.new("RGBA", (width, height), (255, 255, 255, 255))
        frame.thumbnail((width, height), Image.Resampling.LANCZOS)
        canvas.alpha_composite(frame, ((width - frame.width) // 2, (height - frame.height) // 2))
        frames.append(canvas.convert("RGB"))
    return frames


def build_animation_sequence(frames, interval, smooth):
    if not smooth:
        return [(frame, interval) for frame in frames]

    sequence = []
    transition_count = 6
    hold_duration = interval * 0.55
    transition_duration = interval * 0.45 / transition_count
    for index, frame in enumerate(frames):
        next_frame = frames[(index + 1) % len(frames)]
        sequence.append((frame, hold_duration))
        for step in range(1, transition_count + 1):
            amount = step / transition_count
            sequence.append((Image.blend(frame, next_frame, amount), transition_duration))
    return sequence


def source_result(path):
    return {
        "image_id": path.stem,
        "job_id": None,
        "name": path.name,
        "url": f"/api/source/{path.stem}",
    }


@app.post("/api/source/duplicate")
def duplicate_source():
    data = request.json or {}
    image = resolve_image(data.get("image_id"), data.get("job_id"))
    if image is None:
        return jsonify(error="Image not found"), 404

    suffix = image.suffix.lower()
    if suffix not in {f".{ext}" for ext in ALLOWED_IMAGES}:
        suffix = ".png"
    copy_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    shutil.copyfile(image, copy_path)
    return jsonify(image=source_result(copy_path))


@app.post("/api/source/crop")
def crop_source():
    data = request.json or {}
    image = resolve_image(data.get("image_id"), data.get("job_id"))
    if image is None:
        return jsonify(error="Image not found"), 404

    try:
        left = int(data.get("x", 0))
        top = int(data.get("y", 0))
        width = int(data.get("width", 0))
        height = int(data.get("height", 0))
        with Image.open(image) as source:
            source.load()
            left = max(0, min(left, source.width - 1))
            top = max(0, min(top, source.height - 1))
            right = min(source.width, left + width)
            bottom = min(source.height, top + height)
            if right <= left or bottom <= top:
                return jsonify(error="Crop area is empty"), 400
            cropped = source.crop((left, top, right, bottom))
            crop_path = UPLOAD_DIR / f"{uuid.uuid4().hex}.png"
            cropped.save(crop_path, "PNG")
    except (TypeError, ValueError, OSError) as e:
        return jsonify(error=f"Could not crop image: {e}"), 400

    return jsonify(image=source_result(crop_path))


@app.post("/api/remove-background")
def api_remove_background():
    data = request.json or {}
    image_id = data.get("image_id")
    job_id = data.get("job_id")
    image = resolve_image(image_id, job_id)
    if image is None:
        return jsonify(error="Image not found"), 404

    try:
        out = OUTPUT_DIR / f"{uuid.uuid4().hex}.png"
        remove_background(image, out)
        add_history_entry(out.name, "bg", image_id)
        return jsonify(url=f"/api/output/{out.name}")
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.post("/api/segment-click")
def api_segment_click():
    data = request.json or {}
    image_id = data.get("image_id")
    job_id = data.get("job_id")
    points = data.get("points", [])
    image = resolve_image(image_id, job_id)
    if image is None:
        return jsonify(error="Image not found"), 404
    if not points:
        return jsonify(error="No points supplied"), 400

    try:
        result = segment_from_points(image, points)
        mask = result.pop("mask")
        mask_name = f"{uuid.uuid4().hex}.png"
        mask.save(OUTPUT_DIR / mask_name)
        result["mask_url"] = f"/api/output/{mask_name}"
        return jsonify(result)
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.post("/api/export-click")
def api_export_click():
    data = request.json or {}
    image_id = data.get("image_id")
    job_id = data.get("job_id")
    objects = data.get("objects", [])
    image = resolve_image(image_id, job_id)
    if image is None:
        return jsonify(error="Image not found"), 404
    if not objects:
        return jsonify(error="No selected objects"), 400

    try:
        point_lists = [obj.get("points", []) for obj in objects]
        paths = export_click_objects(image, point_lists, OUTPUT_DIR)
        add_history_entries([(p.name, "click", image_id) for p in paths])
        urls = []
        for p in paths:
            urls.append(f"/api/output/{p.name}")
        return jsonify(files=urls)
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.post("/api/extract-all")
def api_extract_all():
    data = request.json or {}
    image_id = data.get("image_id")
    job_id = data.get("job_id")
    chroma_key = bool(data.get("chroma_key", False))
    color_value = str(data.get("chroma_color", "#00ff00"))
    color_match = re.fullmatch(r"#?([0-9a-fA-F]{6})", color_value)
    if color_match:
        rgb = tuple(int(color_match.group(1)[index:index + 2], 16) for index in (0, 2, 4))
        chroma_color = (rgb[2], rgb[1], rgb[0])
    else:
        chroma_color = (0, 255, 0)
    try:
        chroma_tolerance = max(1, min(150, int(data.get("chroma_tolerance", 85))))
        border_thickness = max(0.0, min(0.5, float(data.get("border_thickness", 0))))
    except (TypeError, ValueError):
        return jsonify(error="Invalid chroma or border settings"), 400
    image = resolve_image(image_id, job_id)
    if image is None:
        return jsonify(error="Image not found"), 404

    try:
        objects = extract_all_objects(
            image,
            OUTPUT_DIR,
            chroma_key=chroma_key,
            chroma_color=chroma_color,
            chroma_tolerance=chroma_tolerance,
            border_thickness=border_thickness,
        )
        return jsonify(objects=objects)
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.post("/api/save-selected")
def api_save_selected():
    data = request.json or {}
    files = data.get("files") or []
    preserve_files = data.get("preserve_files") or []
    image_id = data.get("image_id")
    if not files:
        return jsonify(error="No files supplied"), 400

    try:
        valid_files = []
        for file_name in files:
            safe_name = secure_filename(str(file_name))
            if safe_name and (OUTPUT_DIR / safe_name).is_file():
                valid_files.append((safe_name, "auto", image_id))
        if not valid_files:
            return jsonify(error="No valid output files supplied"), 400
        add_history_entries(valid_files, preserve_files=preserve_files)
        return jsonify(ok=True)
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.post("/api/upscale")
def api_upscale():
    data = request.json or {}
    image_id = data.get("image_id")
    job_id = data.get("job_id")
    file_name = data.get("file_name")
    image = data.get("image")
    files = data.get("files") or []
    upscale_mode = str(data.get("mode") or "ai").lower()

    def upscale_source(source_path, source_label):
        output_path = upscale_image(source_path, OUTPUT_DIR, mode=upscale_mode)
        add_history_entry(output_path.name, "upscale", source_label)
        return jsonify(url=f"/api/output/{output_path.name}")

    if image_id:
        source = resolve_image(image_id, job_id)
        if source is None:
            return jsonify(error="Image not found"), 404
        try:
            return upscale_source(source, image_id)
        except Exception as e:
            return jsonify(error=str(e)), 500

    if image is not None:
        safe_name = secure_filename(str(image))
        source = OUTPUT_DIR / safe_name
        if not source.is_file():
            return jsonify(error="Image not found"), 404
        try:
            return upscale_source(source, safe_name)
        except Exception as e:
            return jsonify(error=str(e)), 500

    if file_name:
        safe_name = secure_filename(str(file_name))
        source = (OUTPUT_DIR / safe_name)
        if not source.is_file():
            return jsonify(error="Image not found"), 404
        try:
            return upscale_source(source, safe_name)
        except Exception as e:
            return jsonify(error=str(e)), 500

    if files:
        results = []
        for item in files:
            safe_name = secure_filename(str(item))
            source = OUTPUT_DIR / safe_name
            if not source.is_file():
                continue
            try:
                output_path = upscale_image(source, OUTPUT_DIR, mode=upscale_mode)
                entry = add_history_entry(output_path.name, "upscale", safe_name)
                results.append({"url": f"/api/output/{output_path.name}", "id": entry["id"]})
            except Exception as exc:
                results.append({"error": str(exc), "file": safe_name})
        return jsonify(results=results)

    return jsonify(error="No image supplied"), 400


@app.post("/api/animate")
def api_animate():
    data = request.json or {}
    sources = data.get("sources") or []
    output_format = str(data.get("format") or "gif").lower()
    smooth = bool(data.get("smooth", False))
    try:
        interval = max(0.1, min(2.0, float(data.get("interval", 0.5))))
    except (TypeError, ValueError):
        return jsonify(error="Interval must be between 0.1 and 2 seconds"), 400
    if output_format not in ALLOWED_ANIMATION_FORMATS:
        return jsonify(error="Format must be GIF or MP4"), 400
    if len(sources) < 2:
        return jsonify(error="Add at least two images to the animation"), 400

    try:
        persistent_sources = []
        for source in sources:
            if source.get("file_name"):
                source_path = resolve_animation_source(source)
                if source_path is None:
                    continue
                copy_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{source_path.suffix.lower() or '.png'}"
                shutil.copyfile(source_path, copy_path)
                persistent_sources.append(source_result(copy_path))
            else:
                persistent_sources.append(source)
        frames = build_animation_frames(sources)
        sequence = build_animation_sequence(frames, interval, smooth)
        output_path = OUTPUT_DIR / f"{uuid.uuid4().hex}.{output_format}"
        if output_format == "gif":
            sequence[0][0].save(
                output_path,
                "GIF",
                save_all=True,
                append_images=[frame for frame, _ in sequence[1:]],
                duration=[round(duration * 1000) for _, duration in sequence],
                loop=0,
                optimize=False,
            )
        else:
            video_fps = 20
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                video_fps,
                (sequence[0][0].width, sequence[0][0].height),
            )
            if not writer.isOpened():
                return jsonify(error="MP4 encoding is unavailable on this machine; choose GIF instead."), 500
            for frame, duration in sequence:
                pixels = np.asarray(frame)
                bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
                for _ in range(max(1, round(duration * video_fps))):
                    writer.write(bgr)
            writer.release()

        entry = add_history_entry(output_path.name, "animation", f"{len(frames)} frames")
        return jsonify(
            url=f"/api/output/{output_path.name}",
            format=output_format,
            id=entry["id"],
            sources=persistent_sources,
        )
    except (OSError, ValueError) as e:
        return jsonify(error=f"Could not create animation: {e}"), 400
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.get("/api/output/<name>")
def output(name):
    path = (OUTPUT_DIR / secure_filename(name)).resolve()
    if not str(path).startswith(str(OUTPUT_DIR.resolve())) or not path.is_file():
        abort(404)
    return send_file(path, as_attachment=False)


@app.delete("/api/output/<name>")
def delete_output(name):
    safe_name = secure_filename(name)
    entries = load_history()
    matching = [entry for entry in entries if entry.get("filename") == safe_name]
    if not matching and not (OUTPUT_DIR / safe_name).is_file():
        return jsonify(error="Not found"), 404
    delete_output_file(safe_name)
    save_history([entry for entry in entries if entry.get("filename") != safe_name])
    return jsonify(ok=True)


@app.get("/api/history")
def api_history():
    entries = sorted(load_history(), key=lambda e: e["created_at"], reverse=True)
    return jsonify(history=entries)


@app.delete("/api/history/<entry_id>")
def api_delete_history_entry(entry_id):
    entries = load_history()
    remaining = [e for e in entries if e["id"] != entry_id]
    removed = next((e for e in entries if e["id"] == entry_id), None)
    if removed is None:
        return jsonify(error="Not found"), 404
    delete_output_file(removed.get("filename"))
    save_history(remaining)
    return jsonify(ok=True)


@app.delete("/api/history")
def api_clear_history():
    for e in load_history():
        delete_output_file(e.get("filename"))
    save_history([])
    return jsonify(ok=True)


cleanup_output_dir()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5012)
