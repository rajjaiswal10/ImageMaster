from pathlib import Path
import uuid
import zipfile
import shutil
import json
import time

from flask import Flask, render_template, request, jsonify, send_file, abort
from werkzeug.utils import secure_filename

from processing.background import remove_background
from processing.click_segment import segment_from_points, export_click_objects
from processing.auto_extract import extract_all_objects

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

ALLOWED_IMAGES = {"png", "jpg", "jpeg", "webp", "bmp"}

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


def save_history(entries):
    HISTORY_PATH.write_text(json.dumps(entries, indent=2))
    cleanup_output_dir()


def cleanup_output_dir():
    tracked = {e.get("filename") for e in load_history() if e.get("filename")}
    if not OUTPUT_DIR.exists():
        return
    for item in OUTPUT_DIR.iterdir():
        if item.name == HISTORY_PATH.name:
            continue
        if item.is_file() and item.name not in tracked:
            item.unlink(missing_ok=True)


def cleanup_uploads_on_startup():
    if not UPLOAD_DIR.exists():
        return
    for item in UPLOAD_DIR.iterdir():
        if item.is_dir():
            shutil.rmtree(item, ignore_errors=True)
        elif item.is_file():
            item.unlink(missing_ok=True)


def cleanup_uploads():
    cleanup_uploads_on_startup()


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


def add_history_entries(items):
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
    save_history(entries)
    return added


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/upload")
def upload():
    files = request.files.getlist("files")
    if not files:
        return jsonify(error="No files supplied"), 400

    cleanup_uploads()

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
        urls = []
        for p in paths:
            add_history_entry(p.name, "click", image_id)
            urls.append(f"/api/output/{p.name}")
        return jsonify(files=urls)
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.post("/api/extract-all")
def api_extract_all():
    data = request.json or {}
    image_id = data.get("image_id")
    job_id = data.get("job_id")
    image = resolve_image(image_id, job_id)
    if image is None:
        return jsonify(error="Image not found"), 404

    try:
        objects = extract_all_objects(image, OUTPUT_DIR)
        add_history_entries([
            (o["url"].rsplit("/", 1)[-1], "auto", image_id) for o in objects
        ])
        return jsonify(objects=objects)
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.get("/api/output/<name>")
def output(name):
    path = (OUTPUT_DIR / secure_filename(name)).resolve()
    if not str(path).startswith(str(OUTPUT_DIR.resolve())) or not path.is_file():
        abort(404)
    return send_file(path, as_attachment=False)


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
cleanup_uploads_on_startup()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5012)
