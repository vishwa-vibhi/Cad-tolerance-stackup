"""
Stage 5: Flask Web Dashboard — CAD Tolerance Stack-Up Analysis Tool

Upload a 2D engineering drawing image and get:
  - Annotated visualization with colored overlays
  - Structured annotations table with semantic labels
  - Association visualization (dimension → geometric element)
  - BOM table (Category 2 assembly drawings)
  - Tolerance stack-up summary
  - Part attribution (dimension → part name)
  - Semantic dimension labels (length, height, bore_diameter, etc.)
  - ML classifier results
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

from flask import Flask, render_template, request, jsonify, send_from_directory

# Make src/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vlm_reader       import read_full_image, get_ocr
from validation       import validate_file
from association      import associate_file
from part_attribution import attribute_file
from tolerance_stackup import analyse_file
from semantic_labeller import label_file

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR  = os.path.join(BASE_DIR, 'app', 'uploads')
RESULTS_DIR = os.path.join(BASE_DIR, 'app', 'results')
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# ── Warm up EasyOCR on startup ────────────────────────────────────────────────
print("Warming up EasyOCR...")
try:
    get_ocr()
    print("EasyOCR ready.")
except Exception as e:
    print(f"WARNING: EasyOCR warmup failed: {e}")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def image_to_base64(path):
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        data = f.read()
    ext  = path.rsplit('.', 1)[-1].lower()
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
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file. Upload a PNG or JPG."}), 400

    job_id = str(uuid.uuid4())[:8]
    ext    = file.filename.rsplit('.', 1)[1].lower()

    category_hint = request.form.get('category', 'auto')
    prefix_map    = {'cad1': 'cad1', 'cad2': 'cad2', 'cad3': 'cad3'}
    prefix        = prefix_map.get(category_hint, 'cad1')
    image_filename = f"{prefix}_{job_id}.{ext}"

    image_path = os.path.join(UPLOAD_DIR, image_filename)
    file.save(image_path)

    job_dir  = os.path.join(RESULTS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    basename = os.path.splitext(image_filename)[0]

    results = {"job_id": job_id, "stages": {}}
    t_start = time.time()

    try:
        # ── Stage 2: OCR ──────────────────────────────────────────────────
        print(f"\n[{job_id}] Stage 2: OCR...")
        ocr_results    = read_full_image(image_path, output_dir=job_dir)
        fullocr_path   = os.path.join(job_dir, f"{basename}_fullocr.json")
        high_conf      = sum(1 for r in ocr_results if r['confidence'] > 0.9)
        corrections    = sum(1 for r in ocr_results if r.get('text') != r.get('raw_text'))

        results["stages"]["ocr"] = {
            "detections":   len(ocr_results),
            "high_conf":    high_conf,
            "corrections":  corrections,
            "high_conf_pct": round(high_conf / max(len(ocr_results), 1) * 100, 1),
        }

        # ── Stage 3: Validation + ML classifier ───────────────────────────
        print(f"[{job_id}] Stage 3: Validation + ML...")
        structured      = validate_file(fullocr_path, output_dir=job_dir)
        structured_path = os.path.join(job_dir, f"{basename}_structured.json")

        if structured:
            summary  = structured.get("summary", {})
            total    = structured.get("total_detections", 0)
            unknown  = summary.get("unknown", 0)
            ml_count = sum(1 for e in structured.get("classified", [])
                           if e.get("ml_classified"))
            results["stages"]["validation"] = {
                "total":          total,
                "meaningful":     total - unknown,
                "unknown":        unknown,
                "ml_rescued":     ml_count,
                "meaningful_pct": round((total - unknown) / max(total, 1) * 100, 1),
                "type_summary":   summary,
                "bom_rows":       structured.get("bom_rows", []),
                "classified":     structured.get("classified", []),
                "image_category": structured.get("image_category", 0),
            }

        # ── Stage 4: Association ──────────────────────────────────────────
        print(f"[{job_id}] Stage 4: Association...")
        assoc      = associate_file(image_path, structured_path, output_dir=job_dir)
        assoc_path = os.path.join(job_dir, f"{basename}_associations.json")
        if assoc:
            results["stages"]["association"] = {
                "matched":      assoc.get("matched", 0),
                "unassociated": assoc.get("unassociated", 0),
                "total":        assoc.get("total_annotations", 0),
                "match_pct":    round(assoc.get("matched", 0) /
                                      max(assoc.get("total_annotations", 1), 1) * 100, 1),
            }

        # ── Semantic Labelling ────────────────────────────────────────────
        print(f"[{job_id}] Semantic Labelling...")
        sem_result = None
        if os.path.exists(assoc_path):
            try:
                sem_result = label_file(assoc_path, output_dir=job_dir)
            except Exception as e:
                print(f"  Semantic labelling warning: {e}")

        if sem_result:
            results["stages"]["semantic"] = {
                "labelled":      sem_result.get("labelled_count", 0),
                "total":         sem_result.get("total_annotations", 0),
                "labelled_pct":  sem_result.get("labelled_rate_pct", 0),
                "label_summary": sem_result.get("label_summary", {}),
                "dimension_table": sem_result.get("dimension_table", []),
            }

        # ── Part Attribution ──────────────────────────────────────────────
        print(f"[{job_id}] Part Attribution...")
        attr = attribute_file(structured_path, output_dir=job_dir)
        if attr:
            results["stages"]["attribution"] = {
                "named":         attr.get("named_attributions", 0),
                "total_dims":    attr.get("dimension_annotations", 0),
                "named_pct":     round(attr.get("named_attributions", 0) /
                                       max(attr.get("dimension_annotations", 1), 1) * 100, 1),
                "parts_summary": attr.get("parts_summary", []),
                "confidence_counts": attr.get("confidence_counts", {}),
            }

        # ── Tolerance Stack-Up ────────────────────────────────────────────
        print(f"[{job_id}] Tolerance Stack-Up...")
        stackup = analyse_file(structured_path, output_dir=job_dir)
        if stackup:
            results["stages"]["stackup"] = {
                "dimensions":       stackup["summary"]["total_dimensions"],
                "tolerances":       stackup["summary"]["total_tolerances"],
                "fits":             stackup["summary"]["total_fits"],
                "fit_types":        stackup["summary"]["fit_types_found"],
                "stackup_linear":   stackup.get("stackup_linear"),
                "stackup_diameter": stackup.get("stackup_diameter"),
                "parsed_dims":      stackup.get("dimensions", []),
                "parsed_tols":      stackup.get("tolerances", []),
            }

        results["elapsed_sec"] = round(time.time() - t_start, 1)

        # ── Visualization images ──────────────────────────────────────────
        vis_images = {"original": image_to_base64(image_path)}
        for suffix, key in [('_fullocr.png', 'fullocr'),
                             ('_associations.png', 'associations')]:
            for fname in os.listdir(job_dir):
                if fname.endswith(suffix):
                    vis_images[key] = image_to_base64(os.path.join(job_dir, fname))
                    break

        results["images"]   = vis_images
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
    job_dir = os.path.join(RESULTS_DIR, job_id)
    if not os.path.exists(job_dir):
        return "Not found", 404
    suffix_map = {
        'structured':   '_structured.json',
        'associations': '_associations.json',
        'attributed':   '_attributed.json',
        'stackup':      '_stackup.json',
        'labelled':     '_labelled.json',
    }
    suffix = suffix_map.get(filetype)
    if not suffix:
        return "Not found", 404
    for fname in os.listdir(job_dir):
        if fname.endswith(suffix):
            return send_from_directory(job_dir, fname, as_attachment=True)
    return "File not generated", 404


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  CAD Tolerance Stack-Up Analysis Tool")
    print("  Open: http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(debug=False, host='0.0.0.0', port=5000)
