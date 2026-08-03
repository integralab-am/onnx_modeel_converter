@echo off
title Build FlatVR Model Converter
echo ====================================================
echo      Building FlatVR Model Converter...
echo ====================================================
echo.

REM Try standard MSVC environment first
where cl.exe >nul 2>&1
if %errorlevel% equ 0 (
    echo [INFO] Compiling using active compiler...
    cl /EHsc /std:c++17 model_converter.cpp /Fe:model_converter.exe
) else if exist "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" (
    echo [INFO] Initializing MSVC 2026 x64...
    call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul
    cl /EHsc /std:c++17 model_converter.cpp /Fe:model_converter.exe
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" (
    echo [INFO] Initializing MSVC 2022 x64...
    call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul
    cl /EHsc /std:c++17 model_converter.cpp /Fe:model_converter.exe
) else (
    echo [WARNING] MSVC compiler was not found directly in path.
    echo [INFO] Trying build with CMake...
    mkdir build >nul 2>&1
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    cmake --build . --config Release
    if exist Release\model_converter.exe (
        copy Release\model_converter.exe ..\model_converter.exe >nul
    ) else if exist model_converter.exe (
        copy model_converter.exe ..\model_converter.exe >nul
    )
    cd ..
)

if exist model_converter.obj del model_converter.obj >nul

if exist model_converter.exe (
    echo.
    echo ====================================================
    echo [SUCCESS] model_converter.exe successfully built!
    echo ====================================================
) else (
    echo.
    echo [ERROR] Failed to compile model_converter.exe.
)

pause
