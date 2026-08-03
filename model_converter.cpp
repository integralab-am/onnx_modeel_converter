#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <cstdlib>
#include <windows.h>

namespace fs = std::filesystem;

const std::string PY_SCRIPT_CONTENT = R"(import argparse
import sys
import os

def inline_external_data(path):
    """Fold the weights back into the .onnx file.

    Newer torch exporters write the tensors to a sibling '<name>.onnx.data' and
    leave only the graph behind, which makes the .onnx a ~2 MB stub. FlatVR is
    handed a single path and the instructions say to copy one file, so a model
    split in two silently fails to load wherever it lands.
    """
    try:
        import onnx
    except ImportError:
        print("[WARNING] onnx not available, leaving external weights as-is.")
        return

    try:
        model = onnx.load(path)  # pulls the sidecar in
        onnx.save(model, path, save_as_external_data=False)
    except Exception as e:
        print(f"[WARNING] Could not inline external weights: {e}")
        return

    sidecar = path + ".data"
    if os.path.exists(sidecar):
        try:
            os.remove(sidecar)
        except OSError:
            pass
    print(f"[INFO] Weights inlined; {os.path.basename(path)} is self-contained.")

def export_local_to_onnx(local_dir, resolution=518):
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
    dummy_input = torch.randn(1, 3, resolution, resolution)
    
    # Define output filename based on local folder name
    folder_name = os.path.basename(os.path.normpath(local_dir)).lower().replace("-", "_")
    output_filename = f"{folder_name}_{resolution}.onnx"
    output_path = output_filename
    
    print(f"\nExporting model to FP32 ONNX: {output_path} (Resolution: {resolution}x{resolution})...")
    
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
        
    # Export FP32 ONNX
    try:
        torch.onnx.export(
            wrapper,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['image'],
            output_names=['depth']
        )
        inline_external_data(output_path)
        print(f"[SUCCESS] FP32 ONNX model saved to: {output_path}")
    except Exception as e:
        print(f"[ERROR] ONNX export failed: {e}")
        return False
        
    # Export FP16 ONNX
    try:
        output_filename_fp16 = f"{folder_name}_{resolution}_fp16.onnx"
        output_path_fp16 = output_filename_fp16
        print(f"\nExporting model to FP16 ONNX: {output_path_fp16}...")
        
        model_fp16 = model.half()
        dummy_input_fp16 = dummy_input.half()
        wrapper_fp16 = OnnxWrapper(model_fp16)
        wrapper_fp16.eval()
        
        torch.onnx.export(
            wrapper_fp16,
            dummy_input_fp16,
            output_path_fp16,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['image'],
            output_names=['depth']
        )
        inline_external_data(output_path_fp16)
        print(f"[SUCCESS] FP16 ONNX model saved to: {output_path_fp16}")
    except Exception as e:
        print(f"[WARNING] Could not export FP16 version automatically (often requires GPU): {e}")
        print("You can still use the FP32 version which was successfully exported.")
        
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("local_dir", type=str)
    parser.add_argument("--resolution", type=int, default=518)
    args = parser.parse_args()
    success = export_local_to_onnx(args.local_dir, args.resolution)
    sys.exit(0 if success else 1)
)";

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

    std::string res_str;
    std::cout << "Enter target resolution (e.g. 256, 384, 518) [Default: 518]: ";
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
