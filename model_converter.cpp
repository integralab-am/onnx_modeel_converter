#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <cstdlib>
#include <windows.h>

namespace fs = std::filesystem;

// Split into several literals: MSVC caps a single one at 16 KB.
const std::string PY_SCRIPT_CONTENT = R"PY(import argparse
import json
import sys
import os

# onnxruntime refuses to load anything newer than this, and FlatVR shipped 1.17
# for a long time: "Unsupported model IR version: 10, max supported IR version:
# 9". Current onnx releases stamp 10 by default. Newer FlatVR builds accept 10,
# but 9 still loads on every version anyone has installed.
MAX_IR_VERSION = 9

# Depth Anything 3 is not a transformers model and its package does not install
# with pip on current Python (it caps the version at 3.13 and reads its own
# version from git). Only the network code is needed, so the converter fetches
# the source archive of one known commit and imports it from a folder beside
# itself. Pinned, so an upstream change cannot alter what gets exported.
DA3_COMMIT = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
DA3_ARCHIVE = f"https://github.com/ByteDance-Seed/Depth-Anything-3/archive/{DA3_COMMIT}.zip"
DA3_CODE_DIR = os.path.join("da3_code", DA3_COMMIT[:7])

# Checkpoints under CC BY-NC 4.0. They convert fine; the user should know they
# may not be used commercially. The others (small, base, mono-large,
# metric-large) are Apache 2.0.
DA3_NON_COMMERCIAL = ("da3-large", "da3-giant")

def parse_size(text):
    """'518' -> (518, 518); '518x294' -> (518, 294). Returns (width, height)."""
    text = text.lower().replace(" ", "")
    if "x" in text:
        w, h = text.split("x", 1)
        return int(w), int(h)
    return int(text), int(text)

def da3_model_name(local_dir):
    """The preset name if local_dir holds a Depth Anything 3 checkpoint.

    The Hugging Face DA3 repos carry a config.json that is not a transformers
    config: it names the preset ('da3mono-large') and holds the network spec.
    """
    try:
        with open(os.path.join(local_dir, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        return None
    name = cfg.get("model_name", "")
    if isinstance(name, str) and name.startswith("da3") and "config" in cfg:
        return name
    return None

def finalize_onnx(path, width=None, height=None):
    """Make an exported graph loadable by the runtime FlatVR ships.

    Three things need fixing after export. Newer torch exporters write the
    tensors to a sibling '<name>.onnx.data' and leave the .onnx a ~2 MB stub,
    while the instructions say to copy one file — so what arrives is a graph
    with no weights. The IR version can be stamped higher than the runtime
    accepts, which fails the load outright. And the TorchScript exporter
    leaves the output dimensions unset, which is worth filling in since the
    shape is fixed and known.
    """
    try:
        import onnx
    except ImportError:
        print("[WARNING] onnx not available; leaving the exported file as-is.")
        return

    try:
        model = onnx.load(path)  # pulls the sidecar in
        if model.ir_version > MAX_IR_VERSION:
            print(f"[INFO] Lowering IR version {model.ir_version} -> {MAX_IR_VERSION}.")
            model.ir_version = MAX_IR_VERSION
        if width is not None and height is not None:
            dims = model.graph.output[0].type.tensor_type.shape.dim
            if len(dims) == 3 and all(d.dim_value == 0 for d in dims):
                for d, v in zip(dims, (1, height, width)):
                    d.ClearField('dim_param')
                    d.dim_value = v
        onnx.save(model, path, save_as_external_data=False)
    except Exception as e:
        print(f"[WARNING] Could not finalize {os.path.basename(path)}: {e}")
        return

    sidecar = path + ".data"
    if os.path.exists(sidecar):
        try:
            os.remove(sidecar)
        except OSError:
            pass
    print(f"[INFO] {os.path.basename(path)} is self-contained and IR {MAX_IR_VERSION}.")

)PY"
R"PY(def fetch_da3_code():
    """Folder that contains the depth_anything_3 package, downloading it once."""
    package = os.path.join(DA3_CODE_DIR, "depth_anything_3")
    if os.path.isdir(package):
        return DA3_CODE_DIR
    import io
    import urllib.request
    import zipfile
    print(f"[INFO] Downloading the Depth Anything 3 network code ({DA3_COMMIT[:7]})...")
    with urllib.request.urlopen(DA3_ARCHIVE) as r:
        data = r.read()
    root = f"Depth-Anything-3-{DA3_COMMIT}/"
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for member in z.namelist():
            if member.endswith("/"):
                continue
            if member.startswith(root + "src/depth_anything_3/"):
                rel = member[len(root + "src/"):]
            elif member == root + "LICENSE":
                rel = "LICENSE"
            else:
                continue
            target = os.path.join(DA3_CODE_DIR, *rel.split("/"))
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "wb") as f:
                f.write(z.read(member))
    if not os.path.isdir(package):
        raise RuntimeError("the archive did not contain src/depth_anything_3")
    return DA3_CODE_DIR

def load_da3(local_dir, model_name):
    import torch
    if "nested" in model_name:
        raise RuntimeError(f"{model_name} is two networks in one and is not supported; "
                           "use DA3MONO-LARGE, DA3-SMALL, DA3-BASE or DA3METRIC-LARGE")
    sys.path.insert(0, os.path.abspath(fetch_da3_code()))
    from depth_anything_3.cfg import create_object, load_config
    from depth_anything_3.registry import MODEL_REGISTRY
    from depth_anything_3.model.dinov2.layers import rope
    from safetensors.torch import load_file

    # torch.cartesian_prod has no ONNX export. meshgrid yields the same (y, x)
    # pairs in the same order.
    def positions(self, batch_size, height, width, device):
        yy, xx = torch.meshgrid(torch.arange(height, device=device),
                                torch.arange(width, device=device), indexing="ij")
        pos = torch.stack((yy, xx), dim=-1).reshape(1, height * width, 2)
        return pos.expand(batch_size, -1, -1).clone()
    rope.PositionGetter.__call__ = positions

    if model_name not in MODEL_REGISTRY:
        raise RuntimeError(f"unknown Depth Anything 3 preset '{model_name}'")
    net = create_object(load_config(MODEL_REGISTRY[model_name]))
    weights = os.path.join(local_dir, "model.safetensors")
    state = load_file(weights)
    # The checkpoints wrap the network as 'model.' inside their API class.
    state = {k[len("model."):] if k.startswith("model.") else k: v for k, v in state.items()}
    missing, unexpected = net.load_state_dict(state, strict=False)
    # The multi-view checkpoints leave a few layers of the auxiliary ray head
    # unset. That head is not part of the depth output and is not exported.
    missing = [k for k in missing if "_aux" not in k]
    if missing:
        raise RuntimeError(f"{len(missing)} weights missing, e.g. {missing[:3]}")
    if model_name in DA3_NON_COMMERCIAL:
        print(f"[NOTE] {model_name} is licensed CC BY-NC 4.0: personal, non-commercial use only.")
    return net.eval()

)PY"
R"PY(def make_da3_wrapper(net, half):
    import torch

    class DA3Depth(torch.nn.Module):
        """One image in, FlatVR's map out.

        DA3 takes a batch of views [B, N, 3, H, W] and predicts depth, where
        FlatVR expects Depth Anything V2's convention - nearer is larger - so
        the map is inverted. The sky head is left out on purpose: masking the
        sky to 'infinitely far' also punched holes into foliage and buildings
        it mistook for sky, which looks far worse in 3D than a soft sky.
        """
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, image):
            x = image.unsqueeze(1)
            if half:
                x = x.half()
            h, w = image.shape[-2], image.shape[-1]
            feats, _ = self.net.backbone(x, cam_token=None, export_feat_layers=[])
            out = self.net.head(feats, h, w, patch_start_idx=0)
            depth = out["depth"].float()
            return (1.0 / depth.clamp_min(1e-6)).reshape(1, h, w)

    return DA3Depth(net.half() if half else net).eval()

def export_one(wrapper, dummy_input, path, width, height, opset):
    import torch
    # dynamo=False keeps the old TorchScript exporter. The new one ignores
    # opset_version here (it cannot lower Resize below 18), stamps a newer IR
    # version than onnxruntime 1.17 accepts, and writes the weights to a
    # separate .onnx.data. The old one writes a single file.
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_input,
            path,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=['image'],
            output_names=['depth'],
            dynamo=False
        )
    finalize_onnx(path, width, height)

)PY"
R"PY(def export_da3_to_onnx(local_dir, model_name, width, height):
    import torch
    print("--------------------------------------------------")
    print(f"Loading Depth Anything 3 ({model_name}) from: {local_dir}")
    print("--------------------------------------------------")
    if width % 14 or height % 14:
        print(f"[ERROR] {width}x{height}: both sides must be multiples of 14 for Depth Anything 3.")
        return False
    try:
        net = load_da3(local_dir, model_name)
    except Exception as e:
        print(f"[ERROR] Failed to load the model: {e}")
        return False

    base = output_base_name(local_dir, width, height)
    dummy_input = torch.randn(1, 3, height, width)
    # Opset 17: the attention blocks export as LayerNormalization there, and
    # every onnxruntime FlatVR has shipped supports it.
    try:
        wrapper = make_da3_wrapper(net, half=False)
        with torch.no_grad():
            print(f"Test run successful. Output shape: {list(wrapper(dummy_input).shape)}")
        print(f"\nExporting model to FP32 ONNX: {base}.onnx ({width}x{height})...")
        export_one(wrapper, dummy_input, base + ".onnx", width, height, 17)
        print(f"[SUCCESS] FP32 ONNX model saved to: {base}.onnx")
    except Exception as e:
        print(f"[ERROR] ONNX export failed: {e}")
        return False

    # Same float32 boundary as the transformers path below. On DA3 Mono-Large
    # this one was the faster file: 19 ms against 57 ms as FP32 on DirectML.
    try:
        print(f"\nExporting model to FP16 ONNX: {base}_fp16.onnx...")
        export_one(make_da3_wrapper(net, half=True), dummy_input, base + "_fp16.onnx",
                   width, height, 17)
        print(f"[SUCCESS] FP16 ONNX model saved to: {base}_fp16.onnx")
    except Exception as e:
        print(f"[WARNING] Could not export the FP16 version: {e}")
        print("You can still use the FP32 version which was successfully exported.")
    return True

def output_base_name(local_dir, width, height):
    folder_name = os.path.basename(os.path.normpath(local_dir)).lower().replace("-", "_")
    size = f"{width}" if width == height else f"{width}x{height}"
    return f"{folder_name}_{size}"

)PY"
R"PY(def export_local_to_onnx(local_dir, width=518, height=518):
    print("--------------------------------------------------")
    print(f"Loading local model from directory: {local_dir}")
    print("--------------------------------------------------")
    
    try:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        import torch
    except ImportError:
        print("[ERROR] Required libraries are not installed.")
        print("Please install them using: pip install torch torchvision transformers onnx")
        return False
        
    try:
        # Load local model and processor
        processor = AutoImageProcessor.from_pretrained(local_dir, local_files_only=True)
        model = AutoModelForDepthEstimation.from_pretrained(local_dir, trust_remote_code=True, local_files_only=True)
    except Exception as e:
        print(f"[ERROR] Failed to load local model: {e}")
        return False

    model.eval()
    if width % 14 or height % 14:
        print(f"[WARNING] {width}x{height} is not a multiple of 14 (the DINOv2 patch size); "
              "the edge pixels will be cropped away inside the model.")
    dummy_input = torch.randn(1, 3, height, width)

    # Define output filename based on local folder name
    base = output_base_name(local_dir, width, height)
    output_path = base + ".onnx"

    print(f"\nExporting model to FP32 ONNX: {output_path} ({width}x{height})...")
    
    class OnnxWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, x):
            out = self.model(x)
            if hasattr(out, 'predicted_depth'):
                return out.predicted_depth
            elif isinstance(out, dict) and 'predicted_depth' in out:
                return out['predicted_depth']
            else:
                return out
                
    wrapper = OnnxWrapper(model)
    # The wrapper is built after model.eval(), so it starts out in training
    # mode and the exporter warns about it.
    wrapper.eval()

    # Run test forward pass
    with torch.no_grad():
        try:
            test_out = wrapper(dummy_input)
            print(f"Test run successful. Output shape: {list(test_out.shape)}")
        except Exception as e:
            print(f"[ERROR] Test forward pass failed: {e}")
            return False
        
    # Export FP32 ONNX. Opset 14 / IR 7 in a single file, matching what the
    # stock models ship as.
    try:
        export_one(wrapper, dummy_input, output_path, width, height, 14)
        print(f"[SUCCESS] FP32 ONNX model saved to: {output_path}")
    except Exception as e:
        print(f"[ERROR] ONNX export failed: {e}")
        return False
        
    # Export FP16 ONNX.
    #
    # The boundary has to stay float32: FlatVR binds
    # ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT and fills it from a RWBuffer<float>
    # compute shader, and a float16 input does not raise there, it hangs. So
    # the body is halved and the wrapper casts on the way in and out, letting
    # the exporter place consistent casts itself.
    #
    # Converting the finished FP32 graph with onnxconverter-common was tried
    # and does not work here: keep_io_types left float32 wired straight into an
    # fp16 Conv, and patching that around revealed it had also desynchronised
    # the model's own Cast nodes.
    #
    # Measure before preferring this over the FP32 file. On DirectML the extra
    # casts can push part of the graph onto the CPU, and a Depth Anything V2
    # Small at 392 measured 37 ms as FP16 against 7.5 ms as FP32 on the same
    # GPU. FP16 saves memory; it only saves time when the whole graph still
    # lands on the accelerator.
    try:
        output_path_fp16 = base + "_fp16.onnx"
        print(f"\nExporting model to FP16 ONNX: {output_path_fp16}...")

        class HalfBody(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                out = self.model(x.half())
                if hasattr(out, 'predicted_depth'):
                    out = out.predicted_depth
                elif isinstance(out, dict) and 'predicted_depth' in out:
                    out = out['predicted_depth']
                return out.float()

        wrapper_fp16 = HalfBody(model.half())
        wrapper_fp16.eval()

        export_one(wrapper_fp16, dummy_input, output_path_fp16, width, height, 14)
        print(f"[SUCCESS] FP16 ONNX model saved to: {output_path_fp16}")
    except Exception as e:
        print(f"[WARNING] Could not export the FP16 version: {e}")
        print("You can still use the FP32 version which was successfully exported.")
        
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("local_dir", type=str)
    # One number for a square, or WIDTHxHEIGHT. FlatVR reads the shape off the
    # model, so a 16:9 model spends its patches on the picture, not on bars.
    parser.add_argument("--resolution", type=str, default="518")
    args = parser.parse_args()
    try:
        width, height = parse_size(args.resolution)
    except ValueError:
        print(f"[ERROR] '{args.resolution}' is not a size; use 518 or 518x294.")
        sys.exit(1)
    da3 = da3_model_name(args.local_dir)
    if da3:
        success = export_da3_to_onnx(args.local_dir, da3, width, height)
    else:
        success = export_local_to_onnx(args.local_dir, width, height)
    sys.exit(0 if success else 1)
)PY";

// Depth Anything 3 folders carry a config.json with "model_name": "da3...".
// They need a few packages transformers does not pull in, so the converter has
// to know before it installs anything. The Python side checks the same thing.
bool is_da3_folder(const fs::path& folder) {
    std::ifstream f(folder / "config.json");
    std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    size_t key = text.find("\"model_name\"");
    if (key == std::string::npos) return false;
    size_t open = text.find('"', text.find(':', key) + 1);
    return open != std::string::npos && text.compare(open + 1, 3, "da3") == 0;
}

std::vector<fs::path> find_local_model_folders() {
    std::vector<fs::path> folders;
    fs::path custom_models_dir = ".";
    if (fs::exists(custom_models_dir)) {
        for (const auto& entry : fs::directory_iterator(custom_models_dir)) {
            if (entry.is_directory()) {
                if (fs::exists(entry.path() / "config.json")) {
                    folders.push_back(entry.path());
                }
            }
        }
    }
    return folders;
}

int main() {
    // torch's exporter narrates its progress with emoji. On a console that is
    // not UTF-8 — which is any non-English Windows install — printing one
    // raises 'charmap codec can't encode character', and since that happens on
    // the way out of a finished export, the tool reports failure for work it
    // already did. The child python inherits this environment.
    SetConsoleOutputCP(CP_UTF8);
    _putenv_s("PYTHONIOENCODING", "utf-8");

    std::cout << "====================================================\n";
    std::cout << "      FlatVR Local Depth Model Converter (C++)\n";
    std::cout << "====================================================\n\n";

    // 1. Check Python
    std::cout << "[INFO] Verifying Python installation...\n";
    int check_py = std::system("python --version >nul 2>&1");
    if (check_py != 0) {
        std::cerr << "[ERROR] Python was not found in your PATH. Please install Python 3.11+ first.\n";
        std::cout << "\nPress Enter to exit...";
        std::cin.get();
        return 1;
    }

    // Check for virtual env in project structure
    std::string python_cmd = "python";
    if (fs::exists("../../python/.venv/Scripts/python.exe")) {
        std::cout << "[INFO] Found virtual environment at project root. Using it...\n";
        python_cmd = "..\\..\\python\\.venv\\Scripts\\python.exe";
    } else if (fs::exists("python/.venv/Scripts/python.exe")) {
        std::cout << "[INFO] Found local virtual environment. Using it...\n";
        python_cmd = "python\\.venv\\Scripts\\python.exe";
    }

    // 2. Install dependencies
    std::cout << "[INFO] Verifying/installing packages (torch, transformers, onnx)...\n";
    std::string pip_cmd = python_cmd + " -m pip install torch torchvision transformers onnx --upgrade --no-cache-dir";
    int pip_res = std::system(pip_cmd.c_str());
    if (pip_res != 0) {
        std::cerr << "[ERROR] Failed to install/verify python packages.\n";
        std::cout << "\nPress Enter to exit...";
        std::cin.get();
        return 1;
    }

    // 3. Scan for folders with config.json in the current folder (flatvr/custom_models/)
    std::vector<fs::path> model_folders = find_local_model_folders();

    if (model_folders.empty()) {
        std::cout << "\n====================================================\n";
        std::cout << "[WARNING] No downloaded Hugging Face model folders found.\n";
        std::cout << "Please download the model files from Hugging Face and place\n";
        std::cout << "the folder (containing config.json, model.safetensors, etc.)\n";
        std::cout << "directly inside the 'flatvr/custom_models/' directory.\n";
        std::cout << "====================================================\n";
        std::cout << "\nPress Enter to exit...";
        std::cin.get();
        return 0;
    }

    std::cout << "\n====================================================\n";
    std::cout << "Select local model folder to convert to ONNX:\n";
    for (size_t i = 0; i < model_folders.size(); ++i) {
        std::cout << "  " << (i + 1) << ". " << model_folders[i].filename().string() << "\n";
    }
    std::cout << "====================================================\n\n";

    std::string choice_str;
    std::cout << "Enter folder number (1-" << model_folders.size() << "): ";
    std::getline(std::cin, choice_str);

    int choice = 0;
    try {
        choice = std::stoi(choice_str);
    } catch (...) {}

    if (choice < 1 || choice > (int)model_folders.size()) {
        std::cerr << "[ERROR] Invalid choice.\n";
        std::cout << "\nPress Enter to exit...";
        std::cin.get();
        return 1;
    }

    fs::path selected_folder = model_folders[choice - 1];

    if (is_da3_folder(selected_folder)) {
        std::cout << "[INFO] Depth Anything 3 checkpoint. Installing its extra packages...\n";
        std::string da3_pip = python_cmd + " -m pip install einops addict omegaconf safetensors opencv-python-headless";
        if (std::system(da3_pip.c_str()) != 0) {
            std::cerr << "[ERROR] Failed to install the Depth Anything 3 packages.\n";
            std::cout << "\nPress Enter to exit...";
            std::cin.get();
            return 1;
        }
    }

    std::string res_str;
    std::cout << "Enter target resolution: one number for a square (252, 392, 518)\n"
                 "or WIDTHxHEIGHT for a 16:9 model (518x294, 798x448) [Default: 518]: ";
    std::getline(std::cin, res_str);
    if (res_str.empty()) {
        res_str = "518";
    }

    // Write temp script
    std::string script_path = "temp_export_onnx.py";
    std::ofstream out(script_path);
    if (!out.is_open()) {
        std::cerr << "[ERROR] Cannot create temporary python script.\n";
        return 1;
    }
    out << PY_SCRIPT_CONTENT;
    out.close();

    // Run conversion
    std::cout << "\n[INFO] Starting local conversion...\n";
    std::string run_cmd = python_cmd + " " + script_path + " \"" + selected_folder.string() + "\" --resolution " + res_str;
    int run_res = std::system(run_cmd.c_str());

    fs::remove(script_path);

    if (run_res == 0) {
        std::cout << "\n====================================================\n";
        std::cout << "[SUCCESS] Model successfully converted and saved locally in this folder!\n";
        std::cout << "====================================================\n";
    } else {
        std::cerr << "[ERROR] Model conversion failed. See log output above.\n";
    }

    std::cout << "\nPress Enter to exit...";
    std::cin.get();
    return 0;
}
