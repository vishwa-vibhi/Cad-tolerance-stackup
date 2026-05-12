"""
Stage 5: Flask Web Dashboard — CAD Tolerance Stack-Up Analysis Tool

Upload a 2D engineering drawing image and get:
  - Annotated visualization with colored overlays
  - Structured annotations table (type, text, confidence)
  - Association visualization (dimension → geometric element)
  - BOM table (Category 2 assembly drawings)
  - Tolerance stack-up summary
  - Part attribution (dimension → part name)
  - Download JSON outputs

Run:
    cad_env\\Scripts\\python.exe app/app.py
Then open: http://localhost:5000
"""

import os
import sys
import json
import uuid
import time
import base64
import shutil

from flask import (Flask, render_template, request, jsonify,
                   send_from_directory, url_for)

# Make src/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vlm_reader import read_full_image, get_ocr
from validation import validate_file
from association import associate_file
from part_attribution import attribute_file
from tolerance_stackup import analyse_file

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Directories
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR  = os.path.join(BASE_DIR, 'app', 'uploads')
RESULTS_DIR = os.path.join(BASE_DIR, 'app', 'results')
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# ── Warm up EasyOCR on startup so first request is fast ──────────────────────
print("Warming up EasyOCR (loading models)...")
try:
    get_ocr()
    print("EasyOCR ready.")
except Exception as e:
    print(f"WARNING: EasyOCR warmup failed: {e}")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def image_to_base64(path):
    """Convert image file to base64 string for embedding in HTML."""
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        data = f.read()
    ext = path.rsplit('.', 1)[-1].lower()
    mime = 'image/png' if ext == 'png' else 'image/jpeg'
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


# ============================================================
# Routes
# ============================================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyse', methods=['POST'])
def analyse():
    """
    Main analysis endpoint.
    Accepts an uploaded image, runs the full pipeline, returns results as JSON.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file. Upload a PNG or JPG image."}), 400

    # Save uploaded file with unique ID
    job_id = str(uuid.uuid4())[:8]
    ext = file.filename.rsplit('.', 1)[1].lower()

    # Use category hint to name file so pipeline auto-detects category
    category_hint = request.form.get('category', 'auto')
    if category_hint == 'cad1':
        image_filename = f"cad1_{job_id}.{ext}"
    elif category_hint == 'cad2':
        image_filename = f"cad2_{job_id}.{ext}"
    elif category_hint == 'cad3':
        image_filename = f"cad3_{job_id}.{ext}"
    else:
        image_filename = f"cad1_{job_id}.{ext}"  # default to cat1 for auto

    image_path = os.path.join(UPLOAD_DIR, image_filename)
    file.save(image_path)

    job_dir = os.path.join(RESULTS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    # Derive basename — strip extension
    basename = os.path.splitext(image_filename)[0]
    results = {"job_id": job_id, "stages": {}}
    t_start = time.time()

    try:
        # ── Stage 2: OCR ──────────────────────────────────────────────────
        print(f"\n[{job_id}] Stage 2: OCR...")
        # multipass=False for speed in UI; set True for higher accuracy
        ocr_results = read_full_image(image_path, output_dir=job_dir, multipass=False)
        actual_fullocr = os.path.join(job_dir, f"{basename}_fullocr.json")

        results["stages"]["ocr"] = {
            "detections": len(ocr_results),
            "high_conf": sum(1 for r in ocr_results if r['confidence'] > 0.9),
        }

        # ── Stage 3: Validation ───────────────────────────────────────────
        print(f"[{job_id}] Stage 3: Validation...")
        structured = validate_file(actual_fullocr, output_dir=job_dir)
        structured_path = os.path.join(job_dir, f"{basename}_structured.json")

        if structured:
            summary = structured.get("summary", {})
            results["stages"]["validation"] = {
                "total": structured.get("total_detections", 0),
                "meaningful": structured.get("total_detections", 0) - summary.get("unknown", 0),
                "unknown": summary.get("unknown", 0),
                "type_summary": summary,
                "bom_rows": structured.get("bom_rows", []),
                "classified": structured.get("classified", []),
                "image_category": structured.get("image_category", 0),
            }

        # ── Stage 4: Association ──────────────────────────────────────────
        print(f"[{job_id}] Stage 4: Association...")
        assoc = associate_file(image_path, structured_path, output_dir=job_dir)
        if assoc:
            results["stages"]["association"] = {
                "matched": assoc.get("matched", 0),
                "unassociated": assoc.get("unassociated", 0),
                "total": assoc.get("total_annotations", 0),
            }

        # ── Part Attribution ──────────────────────────────────────────────
        print(f"[{job_id}] Part Attribution...")
        attr = attribute_file(structured_path, output_dir=job_dir)
        if attr:
            results["stages"]["attribution"] = {
                "named": attr.get("named_attributions", 0),
                "total_dims": attr.get("dimension_annotations", 0),
                "parts_summary": attr.get("parts_summary", []),
            }

        # ── Tolerance Stack-Up ────────────────────────────────────────────
        print(f"[{job_id}] Tolerance Stack-Up...")
        stackup = analyse_file(structured_path, output_dir=job_dir)
        if stackup:
            results["stages"]["stackup"] = {
                "dimensions": stackup["summary"]["total_dimensions"],
                "tolerances": stackup["summary"]["total_tolerances"],
                "fits": stackup["summary"]["total_fits"],
                "fit_types": stackup["summary"]["fit_types_found"],
                "stackup_linear": stackup.get("stackup_linear"),
                "stackup_diameter": stackup.get("stackup_diameter"),
                "parsed_dims": stackup.get("dimensions", []),
                "parsed_tols": stackup.get("tolerances", []),
            }

        results["elapsed_sec"] = round(time.time() - t_start, 1)

        # ── Collect visualization images ──────────────────────────────────
        vis_images = {}
        for suffix in ['_fullocr.png', '_associations.png']:
            for fname in os.listdir(job_dir):
                if fname.endswith(suffix):
                    vis_images[suffix.lstrip('_').replace('.png', '')] = \
                        image_to_base64(os.path.join(job_dir, fname))
                    break

        # Original image
        vis_images['original'] = image_to_base64(image_path)
        results["images"] = vis_images
        results["basename"] = basename

        print(f"[{job_id}] Done in {results['elapsed_sec']}s")
        return jsonify(results)

    except Exception as e:
        import traceback
        print(f"[{job_id}] ERROR: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e), "job_id": job_id}), 500


@app.route('/download/<job_id>/<filetype>')
def download(job_id, filetype):
    """Download a result JSON file."""
    job_dir = os.path.join(RESULTS_DIR, job_id)
    if not os.path.exists(job_dir):
        return "Not found", 404
    # Find the file matching the filetype suffix
    suffix_map = {
        'structured':   '_structured.json',
        'associations': '_associations.json',
        'attributed':   '_attributed.json',
        'stackup':      '_stackup.json',
    }
    suffix = suffix_map.get(filetype)
    if not suffix:
        return "Not found", 404
    for fname in os.listdir(job_dir):
        if fname.endswith(suffix):
            return send_from_directory(job_dir, fname, as_attachment=True)
    return "File not generated", 404


if __name__ == '__main__':
    print("\n" + "="*60)
    print("  CAD Tolerance Stack-Up Analysis Tool — Web Dashboard")
    print("  Open: http://localhost:5000")
    print("="*60 + "\n")
    app.run(debug=False, host='0.0.0.0', port=5000)
