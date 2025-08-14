# Android Packaging Guide (with Ultralytics)

This repository provides a Tkinter desktop app. Tkinter does not run on Android. To get this project onto Android, there are two practical paths depending on your goals:

- Option A (recommended): Build a native Android app (Kotlin/Java) and run an exported Ultralytics model on‑device (TFLite/ONNX/NCNN). Use the Python code here mainly as reference for business logic, and move the UI to Android. This gives the best performance and UX.
- Option B (advanced/experimental): Port the Python app to Kivy and package with Buildozer (Python‑for‑Android). You must avoid PyTorch (Ultralytics runtime) on device and instead use exported models (e.g., TFLite) via `tflite-runtime`. OpenCV and camera support on Android with Python has constraints. This path requires code changes and careful dependency management.

Below are detailed steps for both options, plus an optional remote‑processing approach.

---

## Option A: Native Android app with exported Ultralytics model (recommended)

Ultralytics models (like `yolov8n-pose.pt`) can be exported to formats suitable for mobile. Then you load and run them in a native Android project.

### 1) Export the model from this repo

We added a helper script `export_model.py` to convert the included `yolov8n-pose.pt` into TFLite and ONNX. From the project root:

```bash
# Create a venv (recommended)
python -m venv .venv && source .venv/bin/activate
pip install ultralytics

# Export to TFLite and ONNX (outputs into ./models)
python export_model.py --weights yolov8n-pose.pt --formats tflite onnx
```

Outputs (examples):
- `models/yolov8n-pose_fp32.tflite`
- `models/yolov8n-pose_dynamic.onnx`

Notes:
- For smaller, faster models, consider INT8 or FP16 quantization. See `python export_model.py --help` for options (e.g., `--int8`, `--fp16`). INT8 requires a calibration set.
- You can also export to `ncnn`, `onnx`, or `openvino` depending on your Android runtime choice.

### 2) Create an Android Studio project

- Minimum SDK: 24+ (Android 7.0) recommended.
- Add camera permissions to `AndroidManifest.xml`:

```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-feature android:name="android.hardware.camera.any" />
```

- Place your exported model in `app/src/main/assets/` (e.g., `app/src/main/assets/yolov8n-pose_fp32.tflite`).

### 3) Choose a runtime and add dependencies

Common choices:
- TensorFlow Lite: simple integration, broad support.
- ONNX Runtime Mobile: flexible, supports many ops.
- NCNN/MNN: fast and lightweight C++ runtimes for mobile.

Example: TensorFlow Lite (Gradle):

```gradle
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:2.14.0'
    // If your model has custom ops:
    // implementation 'org.tensorflow:tensorflow-lite-select-tf-ops:2.14.0'
}
```

### 4) Run inference (Kotlin, TFLite example)

```kotlin
class PoseDetector(context: Context) {
    private val tflite: Interpreter

    init {
        val model = FileUtil.loadMappedFile(context, "yolov8n-pose_fp32.tflite")
        val options = Interpreter.Options().apply {
            // Enable NNAPI or GPU delegate if available
            // addDelegate(NnApiDelegate())
        }
        tflite = Interpreter(model, options)
    }

    fun infer(inputTensor: FloatArray, inputShape: IntArray): FloatArray {
        // Prepare inputs/outputs according to your exported model's signature
        val outputBuffer = Array(1) { FloatArray(56 * 8400) } // example, adjust to your model
        tflite.run(inputTensor, outputBuffer)
        return outputBuffer[0]
    }
}
```

- Preprocess the camera frame to match model input (size, normalization, letterbox) just like Ultralytics does. After inference, decode boxes/keypoints and draw overlays. Ultralytics docs provide guidance on output decoding for YOLOv8/v10 pose.
- For a head start, consult the Ultralytics Android examples:
  - https://docs.ultralytics.com/integrations/android/
  - https://github.com/ultralytics/ultralytics/tree/main/examples/android

### 5) Performance tips
- Prefer small models (e.g., `yolov8n-pose`) and FP16/INT8 if quality is acceptable.
- Use GPU or NNAPI delegates.
- Keep image sizes modest (e.g., 416–640 px max side).
- Avoid unnecessary copies; use `ImageReader`/`CameraX` wisely.

---

## Option B: Python-on-Android with Kivy + Buildozer (advanced)

This app uses Tkinter, which is not supported on Android. To package a Python app on Android you typically:

1) Port UI to Kivy (or another Android‑friendly Python framework). Replace all Tkinter code with Kivy widgets and event loop.
2) Use Buildozer (Python‑for‑Android) to build an APK.
3) Avoid PyTorch on device; use an exported model with `tflite-runtime` or a supported runtime recipe.

High‑level steps:
- Refactor `main.py` into Kivy architecture (e.g., `main_kivy.py`).
- Use `Camera` or `android-camera` in Kivy; OpenCV via p4a is possible but tricky.
- Add a `buildozer.spec` with requirements like `kivy, tflite-runtime, numpy` (avoid `ultralytics` to skip PyTorch). Handle permissions (`android.permissions = CAMERA`).
- Write a small Python wrapper to load the `.tflite` with `tflite_runtime.Interpreter` and run inference on frames.

Caveats:
- OpenCV GUI windows and Tkinter do not work on Android. Use Kivy canvas for drawing overlays.
- Packaging PyTorch with p4a is not supported; that’s why we export the model.
- Camera and performance tuning require device testing.

If you plan to go this route, it’s a separate development effort beyond this repository’s scope, but the above constraints and steps should save time.

---

## Option C: Remote processing (simplest to ship Python)

- Run this Python app on a server/desktop.
- Build a thin Android viewer app that streams the camera to the Python service (WebRTC/RTSP/HTTP) and receives back overlays/metrics.
- Good when devices are low‑power but you control a local network.

---

## FAQ

- Can I package this exact Tkinter app for Android? No. Tkinter is not supported on Android. You must either go native (Android Studio) or switch to an Android‑friendly Python UI framework (Kivy/Buildozer) and drop PyTorch.
- Can I run Ultralytics directly on Android with Python? Not with PyTorch easily. Use `ultralytics export` to TFLite/ONNX/etc., and run those with mobile runtimes.
- Where do I start? If you want a real Android app quickly, use Option A. If you want to stay in Python, budget time to port to Kivy and export the model.

---

## Commands quick reference

```bash
# Export model for Android use
python -m venv .venv && source .venv/bin/activate
pip install ultralytics
python export_model.py --weights yolov8n-pose.pt --formats tflite onnx --fp16

# Result models appear in ./models/
ls models/
```
