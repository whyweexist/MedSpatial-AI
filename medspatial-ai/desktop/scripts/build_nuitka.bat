@echo off
:: ============================================================================
:: MedSpatial AI — Nuitka Production Compile Script
:: ============================================================================
:: Compiles the production FastAPI backend into a single self-contained .exe
:: using Nuitka. No Python installation required on the target machine.
::
:: Prerequisites (install once on the build machine / GitHub runner):
::   pip install nuitka==2.3.0 ordered-set zstandard
::   pip install -r desktop/backend_prod/requirements_prod.txt
::
:: Output:
::   desktop/backend_prod_dist/medspatial_backend.exe  (~120-160 MB)
::
:: Usage:
::   cd medspatial-ai
::   desktop\scripts\build_nuitka.bat
::
:: Key decisions:
::   --nofollow-import-to=torch          torch is NEVER included (ONNX replaces it)
::   --nofollow-import-to=pydantic_settings  replaced by config_prod.py plain class
::   matplotlib IS included: report_service._render_slices_to_images() uses it
::   tokenizers/huggingface_hub ARE included: used by transformers tokenizer
::   sentence_transformers IS included: used by chat service
::   pydantic_settings is NOT included: replaced by config_prod.py
:: ============================================================================

setlocal EnableDelayedExpansion

:: ── Paths ────────────────────────────────────────────────────────────────────
set "REPO_ROOT=%~dp0..\.."
set "BACKEND_PROD=%REPO_ROOT%\desktop\backend_prod"
set "BACKEND_SRC=%REPO_ROOT%\backend"
set "OUT_DIR=%REPO_ROOT%\desktop\backend_prod_dist"
set "BUILD_WORK=%REPO_ROOT%\desktop\_nuitka_build_work"
set "ENTRY=%BACKEND_PROD%\main_prod.py"

:: ── Validate Python ───────────────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.11 and add to PATH.
    exit /b 1
)

:: ── Validate Nuitka ───────────────────────────────────────────────────────────
python -m nuitka --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Nuitka not found. Run: pip install nuitka==2.3.0 ordered-set zstandard
    exit /b 1
)

echo.
echo ============================================================
echo  MedSpatial AI — Nuitka Backend Compile
echo  Entry   : %ENTRY%
echo  Output  : %OUT_DIR%
echo ============================================================
echo.

:: ── Create output directory ───────────────────────────────────────────────────
if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"
if not exist "%BUILD_WORK%" mkdir "%BUILD_WORK%"

:: ── Run Nuitka ────────────────────────────────────────────────────────────────
python -m nuitka ^
    --standalone ^
    --onefile ^
    --onefile-no-splash ^
    --output-dir="%OUT_DIR%" ^
    --output-filename=medspatial_backend.exe ^
    --windows-console-mode=disable ^
    --windows-product-name="MedSpatial AI Backend" ^
    --windows-product-version=1.0.0.0 ^
    --windows-file-description="MedSpatial AI FastAPI Backend" ^
    --windows-company-name="MedSpatial AI" ^
    ^
    --python-flag=no_site ^
    --python-flag=no_warnings ^
    ^
    --follow-imports ^
    --follow-import-to=app ^
    --follow-import-to=fastapi ^
    --follow-import-to=uvicorn ^
    --follow-import-to=starlette ^
    --follow-import-to=pydantic ^
    --follow-import-to=aiosqlite ^
    --follow-import-to=sqlalchemy ^
    --follow-import-to=loguru ^
    --follow-import-to=pydicom ^
    --follow-import-to=scipy ^
    --follow-import-to=sklearn ^
    --follow-import-to=skimage ^
    --follow-import-to=PIL ^
    --follow-import-to=trimesh ^
    --follow-import-to=reportlab ^
    --follow-import-to=docx ^
    --follow-import-to=onnxruntime ^
    --follow-import-to=numpy ^
    --follow-import-to=einops ^
    --follow-import-to=matplotlib ^
    --follow-import-to=transformers ^
    --follow-import-to=tokenizers ^
    --follow-import-to=huggingface_hub ^
    --follow-import-to=sentence_transformers ^
    ^
    --include-package=app ^
    --include-package=app.ai ^
    --include-package=app.api ^
    --include-package=app.core ^
    --include-package=app.models ^
    --include-package=app.services ^
    --include-package=app.schemas ^
    --include-package=app.atlas ^
    --include-package=app.knowledge ^
    --include-package=fastapi ^
    --include-package=uvicorn ^
    --include-package=starlette ^
    --include-package=pydantic ^
    --include-package=sqlalchemy ^
    --include-package=aiosqlite ^
    --include-package=onnxruntime ^
    --include-package=pydicom ^
    --include-package=scipy ^
    --include-package=skimage ^
    --include-package=sklearn ^
    --include-package=PIL ^
    --include-package=trimesh ^
    --include-package=reportlab ^
    --include-package=docx ^
    --include-package=loguru ^
    --include-package=einops ^
    --include-package=matplotlib ^
    --include-package=transformers ^
    --include-package=tokenizers ^
    --include-package=huggingface_hub ^
    --include-package=sentence_transformers ^
    ^
    --include-package-data=onnxruntime ^
    --include-package-data=pydicom ^
    --include-package-data=trimesh ^
    --include-package-data=reportlab ^
    --include-package-data=sklearn ^
    --include-package-data=matplotlib ^
    --include-package-data=transformers ^
    --include-package-data=sentence_transformers ^
    ^
    --nofollow-import-to=torch ^
    --nofollow-import-to=torchvision ^
    --nofollow-import-to=torchaudio ^
    --nofollow-import-to=tensorboard ^
    --nofollow-import-to=tensorflow ^
    --nofollow-import-to=jax ^
    --nofollow-import-to=pytest ^
    --nofollow-import-to=IPython ^
    --nofollow-import-to=notebook ^
    --nofollow-import-to=jupyter ^
    --nofollow-import-to=pydantic_settings ^
    --nofollow-import-to=tkinter ^
    --nofollow-import-to=wx ^
    --nofollow-import-to=PyQt5 ^
    --nofollow-import-to=PyQt6 ^
    --nofollow-import-to=gi ^
    --nofollow-import-to=gtk ^
    ^
    --enable-plugin=anti-bloat ^
    --enable-plugin=numpy ^
    --enable-plugin=pylint-warnings ^
    ^
    --remove-output ^
    ^
    --include-data-dir="%BACKEND_SRC%\app"=app ^
    --include-data-files="%BACKEND_PROD%\onnx_inference.py"=onnx_inference.py ^
    --include-data-files="%BACKEND_PROD%\config_prod.py"=config_prod.py ^
    ^
    "%ENTRY%"

if errorlevel 1 (
    echo.
    echo [ERROR] Nuitka compilation FAILED. See output above.
    exit /b 1
)

:: ── Verify output ─────────────────────────────────────────────────────────────
if not exist "%OUT_DIR%\medspatial_backend.exe" (
    echo [ERROR] Expected output not found: %OUT_DIR%\medspatial_backend.exe
    exit /b 1
)

:: ── Report size ───────────────────────────────────────────────────────────────
for %%F in ("%OUT_DIR%\medspatial_backend.exe") do (
    set /a SIZE_MB=%%~zF / 1048576
    echo.
    echo [OK] Compiled: %OUT_DIR%\medspatial_backend.exe
    echo [OK] Size    : !SIZE_MB! MB
)

:: ── Copy sidecar into Tauri src-tauri directory ───────────────────────────────
:: Tauri expects the binary at  src-tauri/<name>-<target-triple>.exe
:: The target triple for Windows x64 is: x86_64-pc-windows-msvc
set "TAURI_BIN_DIR=%REPO_ROOT%\desktop\tauri\src-tauri"
set "TRIPLE=x86_64-pc-windows-msvc"

copy /Y "%OUT_DIR%\medspatial_backend.exe" ^
       "%TAURI_BIN_DIR%\medspatial_backend-%TRIPLE%.exe" >nul

echo [OK] Sidecar copied → %TAURI_BIN_DIR%\medspatial_backend-%TRIPLE%.exe
echo.
echo ============================================================
echo  Nuitka build COMPLETE.
echo  Next step: run build_all.bat or tauri build in desktop\tauri\
echo ============================================================

endlocal
exit /b 0
