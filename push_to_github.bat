@echo off
title Push to GitHub
echo ====================================================
echo      Pushing Model Converter to GitHub...
echo ====================================================
echo.

if not exist .git (
    echo [INFO] Initializing Git repository...
    git init
    git branch -M main
)

echo [INFO] Staging files...
git add .

echo [INFO] Creating commit...
git commit -m "Initial commit of FlatVR Model Converter"

echo [INFO] Setting remote origin...
git remote remove origin >nul 2>&1
git remote add origin https://github.com/integralab-am/onnx_modeel_converter.git

echo.
echo ====================================================
echo [IMPORTANT] Git will now attempt to push.
echo If prompted, please authenticate with your GitHub account.
echo ====================================================
echo.

git push -u origin main

if %errorlevel% eq 0 (
    echo.
    echo [SUCCESS] Successfully pushed to GitHub!
) else (
    echo.
    echo [ERROR] Failed to push to GitHub. Please check credentials/permissions.
)

pause
