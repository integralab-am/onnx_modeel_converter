# FlatVR Depth Model Converter

This tool is designed to locally convert raw PyTorch depth estimation models (such as **Depth Anything V2**, **Distill-Any-Depth**, and **DinoV2**) from Hugging Face into the optimized **ONNX** format required by **FlatVR**.

Models are exported with a fixed resolution and unified input/output tensor configurations:
* **Input Node:** `image` (shape `[1, 3, Resolution, Resolution]`, **float32**, normalized using ImageNet mean & std dev)
* **Output Node:** `depth` (shape `[1, Resolution, Resolution]`, **float32**)

The tool automatically exports both the high-precision **FP32** version (`.onnx`) and the performance-optimized **FP16** version (`_fp16.onnx`).

In the FP16 build only the weights are half precision — the graph boundary stays float32. FlatVR binds float32 tensors and fills them from a `RWBuffer<float>` compute shader, so a model with float16 inputs will not raise an error, it will hang. Both files are self-contained; the weights are never left in a separate `.onnx.data` sidecar.

> **Measure before assuming FP16 is faster.** The casts it needs can leave part of the graph off the accelerator. Depth Anything V2 Small at 392×392 measured **37 ms as FP16 against 7.5 ms as FP32** on the same GPU under DirectML, because the FP16 graph did not fully assign to the DirectML provider. FP16 reliably saves memory; it saves time only when the whole graph still lands on the accelerator. Try both.

Pick a resolution that is a **multiple of 14**, the patch size of the DINOv2 backbone (e.g. `252`, `336`, `392`, `448`, `518`). Cost scales with the number of patches, so 392 sits roughly midway between the stock 252 and 518 presets in both speed and quality.

---

## 🚀 How to Use

### 1. Download the Model from Hugging Face
Download or `git clone` the model repository folder from Hugging Face and place it in the same directory as `model_converter.exe`. 
The folder must contain the following configuration and weight files:
* `config.json`
* `model.safetensors` (or `pytorch_model.bin`)
* `preprocessor_config.json`

For example:
```
model_converter.exe
├─ depth-anything-v2-small-hf/
│  ├─ config.json
│  ├─ model.safetensors
│  └─ preprocessor_config.json
```

### 2. Run the Converter
Double-click `model_converter.exe`:
1. The utility will verify that Python is installed.
2. It scans the directory and lists all discovered local Hugging Face model folders.
3. Select the folder by entering its number.
4. Specify your desired target resolution (e.g. `256`, `384`, or `518`).
5. The script installs required libraries (`torch`, `transformers`, `onnx`) in a local virtual environment and runs the export process.

Once complete, the `.onnx` and `_fp16.onnx` files will be saved in the directory. Simply copy them to the `flatvr/custom_models/` folder in your FlatVR installation path.

---

## 🛠️ Building from Source

### Requirements
* A C++17 compatible compiler (MSVC / GCC / Clang).
* *Optional:* CMake 3.15+.

### Build on Windows (MSVC)
Run the `build.bat` script. It will automatically detect your Visual Studio compiler environment and compile the executable.

Alternatively, compile manually from a Developer Command Prompt:
```bash
cl /EHsc /std:c++17 model_converter.cpp /Fe:model_converter.exe
```
