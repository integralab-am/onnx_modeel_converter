# FlatVR Depth Model Converter

This tool is designed to locally convert raw PyTorch depth estimation models (such as **Depth Anything V2**, **Distill-Any-Depth**, and **DinoV2**) from Hugging Face into the optimized **ONNX** format required by **FlatVR**.

Models are exported with a fixed resolution and unified input/output tensor configurations:
* **Input Node:** `image` (shape `[1, 3, Resolution, Resolution]`, normalized using ImageNet mean & std dev)
* **Output Node:** `depth` (shape `[1, Resolution, Resolution]`)

The tool automatically exports both the high-precision **FP32** version (`.onnx`) and the performance-optimized **FP16** version (`_fp16.onnx`).

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
