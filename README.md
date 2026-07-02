# FlatVR Depth Model Converter

Этот инструмент предназначен для локальной конвертации сырых PyTorch-моделей оценки глубины (например, **Depth Anything V2**, **Distill-Any-Depth**, **DinoV2**) с Hugging Face в оптимизированный формат **ONNX** для использования во **FlatVR**.

Модели преобразуются в структуру с фиксированным разрешением и единым входом/выходом:
* **Входной порт:** `image` (размерность `[1, 3, Resolution, Resolution]`, нормализованный по схеме ImageNet)
* **Выходной порт:** `depth` (размерность `[1, Resolution, Resolution]`)

Экспортируются сразу две версии модели: стандартная **FP32** (`.onnx`) и ускоренная **FP16** (`_fp16.onnx`).

---

## 🚀 Как пользоваться

### 1. Скачайте модель с Hugging Face
Скачайте репозиторий модели целиком с Hugging Face с помощью браузера или команды `git clone` и положите эту папку рядом с исполняемым файлом `model_converter.exe`.
Папка модели обязательно должна содержать файлы структуры и весов:
* `config.json`
* `model.safetensors` (или `pytorch_model.bin`)
* `preprocessor_config.json`

Например, структура должна выглядеть так:
```
model_converter.exe
├─ depth-anything-v2-small-hf/
│  ├─ config.json
│  ├─ model.safetensors
│  └─ preprocessor_config.json
```

### 2. Запустите конвертер
Запустите `model_converter.exe`:
1. Утилита проверит наличие Python на компьютере.
2. Просканирует текущую папку и выведет список всех найденных локальных папок моделей.
3. Попросит ввести номер нужной папки.
4. Попросит указать желаемое разрешение разрешения (например, `256`, `384` или `518`).
5. Проверит/установит необходимые библиотеки PyTorch/Transformers и выполнит локальную сборку ONNX.

Полученные файлы `.onnx` и `_fp16.onnx` появятся в этой же папке. Просто перенесите их в папку `flatvr/custom_models/` в директории FlatVR.

---

## 🛠️ Сборка утилиты из исходного кода

### Требования
* Компилятор C++ с поддержкой стандарта C++17 (MSVC / GCC / Clang).
* *Необязательно:* Установленный CMake 3.15+.

### Сборка на Windows (MSVC)
Просто запустите файл `build.bat` двойным кликом. Скрипт попытается автоматически найти Visual Studio и скомпилировать исполняемый файл.

Или выполните в Developer Command Prompt:
```bash
cl /EHsc /std:c++17 model_converter.cpp /Fe:model_converter.exe
```

---

## English Quick Start

1. Download raw Hugging Face model folder (containing `config.json` and `model.safetensors`) and place it next to `model_converter.exe`.
2. Run `model_converter.exe`.
3. Choose the folder and target resolution.
4. The tool will export optimized `model_518.onnx` and `model_518_fp16.onnx` files locally.
5. Move these files to your `flatvr/custom_models/` directory.

To build the utility, run `build.bat` or compile manually:
```bash
cl /EHsc /std:c++17 model_converter.cpp /Fe:model_converter.exe
```
