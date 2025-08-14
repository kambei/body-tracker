#!/usr/bin/env python3
"""
Export an Ultralytics YOLO pose model to mobile-friendly formats for Android.

Examples:
  python export_model.py --weights yolov8n-pose.pt --formats tflite onnx
  python export_model.py --weights yolov8n-pose.pt --formats tflite --fp16
  python export_model.py --weights yolov8n-pose.pt --formats tflite --int8 --calib ./calib_images

Notes:
- Requires `ultralytics` installed. Install via: pip install ultralytics
- Outputs go into ./models/
- INT8 export requires representative dataset for calibration.
"""
import argparse
import os
from pathlib import Path

try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None
    _IMPORT_ERROR = str(e)


def export_tflite(model: 'YOLO', out_dir: Path, fp16: bool, int8: bool, calib_dir: Path | None):
    kwargs = {"format": "tflite"}
    suffix = "fp32"
    if fp16 and int8:
        raise ValueError("Choose only one of --fp16 or --int8 for TFLite export.")
    if fp16:
        kwargs["half"] = True
        suffix = "fp16"
    if int8:
        # INT8 requires a representative dataset
        if calib_dir is None or not calib_dir.exists():
            raise ValueError("INT8 quantization requires --calib pointing to a directory with images.")
        # Ultralytics uses `data` or `int8` with representative dataset; implementation may change across versions.
        # We'll pass in the directory via `int8` argument.
        kwargs["int8"] = True
        kwargs["calib"] = str(calib_dir)
        suffix = "int8"
    print(f"[export] TFLite ({suffix}) ...")
    results = model.export(**kwargs)
    # results returns path; copy/rename into out_dir with suffix for clarity
    src = Path(results)
    dst = out_dir / f"{model.model_name or 'model'}_{suffix}.tflite"
    dst.write_bytes(src.read_bytes())
    print(f"[ok] {dst}")


def export_onnx(model: 'YOLO', out_dir: Path, dynamic: bool):
    print("[export] ONNX ...")
    results = model.export(format="onnx", dynamic=dynamic)
    src = Path(results)
    dst = out_dir / f"{model.model_name or 'model'}_{'dynamic' if dynamic else 'static'}.onnx"
    dst.write_bytes(src.read_bytes())
    print(f"[ok] {dst}")


def main():
    parser = argparse.ArgumentParser(description="Export Ultralytics model for Android")
    parser.add_argument("--weights", type=str, default="yolov8n-pose.pt", help="Path to .pt weights")
    parser.add_argument("--formats", nargs="+", choices=["tflite", "onnx", "ncnn"], default=["tflite"], help="Formats to export")
    parser.add_argument("--out", type=str, default="models", help="Output directory")
    parser.add_argument("--fp16", action="store_true", help="Export half precision where supported (e.g., TFLite)")
    parser.add_argument("--int8", action="store_true", help="INT8 quantization (requires calibration images)")
    parser.add_argument("--calib", type=str, default=None, help="Directory with calibration images for INT8")
    parser.add_argument("--onnx-dynamic", action="store_true", help="Export ONNX with dynamic axes")
    args = parser.parse_args()

    if YOLO is None:
        raise SystemExit(f"Ultralytics import failed: {_IMPORT_ERROR}\nInstall with: pip install ultralytics")

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise SystemExit(f"Weights not found: {weights_path}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {weights_path}")
    model = YOLO(str(weights_path))

    # Persist model name for nicer filenames
    try:
        model.model_name = weights_path.stem
    except Exception:
        model.model_name = None

    if "tflite" in args.formats:
        export_tflite(model, out_dir, fp16=args.fp16, int8=args.int8, calib_dir=Path(args.calib) if args.calib else None)

    if "onnx" in args.formats:
        export_onnx(model, out_dir, dynamic=args.onnx_dynamic)

    if "ncnn" in args.formats:
        print("[export] NCNN ...")
        results = model.export(format="ncnn")
        srcdir = Path(results)
        # NCNN creates a directory with .param and .bin
        target_dir = out_dir / f"{model.model_name or 'model'}_ncnn"
        target_dir.mkdir(exist_ok=True)
        for item in srcdir.iterdir():
            (target_dir / item.name).write_bytes(item.read_bytes())
        print(f"[ok] {target_dir}")

    print("[done] Exports complete.")


if __name__ == "__main__":
    main()
