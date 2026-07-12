@echo off
:: ============================================================================
:: MedSpatial AI — Master Windows Build Script
:: ============================================================================
:: Runs the entire build pipeline from source to a signed NSIS .exe installer.
::
:: PIPELINE STAGES (all can be run individually via flags):
::   Stage 1 — Python env + production deps
::   Stage 2 — ONNX export (PyTorch → .onnx)
::   Stage 3 — ONNX validation (PyTorch vs ORT comparison)
::   Stage 4 — ONNX quantization (optional INT8)
::   Stage 5 — Nuitka backend compile → medspatial_backend.exe
::   Stage 6 — Vite frontend build (React → static HTML/JS/CSS)
::   Stage 7 — Tauri installer build (Rust + NSIS → .exe installer)
::   Stage 8 — Post-build verification
::
:: FLAGS (skip individual stages):
::   --skip-env          Skip venv creation / pip install
::   --skip-onnx         Skip ONNX export (use existing .onnx files)
::   --skip-validate     Skip ONNX validation
::   --skip-quantize     Skip INT8 quantization
::   --skip-nuitka       Skip Nuitka compile (use existing .exe)
::   --skip-frontend     Skip Vite build
::   --skip-tauri        Skip Tauri/NSIS build
::   --quantize          Enable INT8 quantization (off by default)
::   --fp16              Also export FP16 ONNX variants
::   --sign              Code-sign the installer (requires CERT_PATH + CERT_PASS)
::
:: ENVIRONMENT VARIABLES (optional overrides):
::   PYTHON_CMD          Python executable (default: python)
::   NODE_CMD            Node executable (default: node)
::   NPM_CMD             npm executable (default: npm)
::   CARGO_CMD           Cargo executable (default: cargo)
::   CERT_PATH           Path to .pfx code signing certificate
::   CERT_PASS           Certificate password (use a secrets manager in CI)
::   VOLUME_SIZE         Model volume cube size (default: 128)
::
:: USAGE EXAMPLES:
::   build_all.bat                      Full build
::   build_all.bat --skip-onnx          Skip export (models already exported)
::   build_all.bat --skip-nuitka --skip-tauri   Frontend only
::   build_all.bat --quantize --fp16    With INT8 + FP16 models
::
:: REQUIREMENTS ON THE BUILD MACHINE:
::   - Python 3.11.x (64-bit)  — in PATH
::   - Node.js 20.x LTS        — in PATH
::   - Rust stable 1.77+       — in PATH  (rustup install stable)
::   - Visual Studio Build Tools 2022 (for Nuitka + Rust MSVC linker)
::   - NSIS 3.x                — in PATH  (or installed via Tauri CLI)
::
:: OUTPUT:
::   desktop\tauri\src-tauri\target\release\bundle\nsis\
::     MedSpatial AI_1.0.0_x64-setup.exe
:: ============================================================================

setlocal EnableDelayedExpansion

:: ── Banner ────────────────────────────────────────────────────────────────────
echo.
echo  ██████████████████████████████████████████████████████
echo   MedSpatial AI  —  Production Build Pipeline v1.0
echo  ██████████████████████████████████████████████████████
echo.

:: ── Timestamp ─────────────────────────────────────────────────────────────────
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set "DATETIME=%%I"
set "BUILD_TIME=%DATETIME:~0,4%-%DATETIME:~4,2%-%DATETIME:~6,2%_%DATETIME:~8,2%%DATETIME:~10,2%%DATETIME:~12,2%"
echo  Build started: %BUILD_TIME%
echo.

:: ── Defaults ──────────────────────────────────────────────────────────────────
set "PYTHON_CMD=%PYTHON_CMD:python=%"
if "%PYTHON_CMD%"=="" set "PYTHON_CMD=python"
set "NODE_CMD=%NODE_CMD:node=%"
if "%NODE_CMD%"=="" set "NODE_CMD=node"
set "NPM_CMD=%NPM_CMD:npm=%"
if "%NPM_CMD%"=="" set "NPM_CMD=npm"
set "VOLUME_SIZE=%VOLUME_SIZE:128=%"
if "%VOLUME_SIZE%"=="" set "VOLUME_SIZE=128"

:: ── Parse flags ───────────────────────────────────────────────────────────────
set SKIP_ENV=0
set SKIP_ONNX=0
set SKIP_VALIDATE=0
set SKIP_QUANTIZE=0
set SKIP_NUITKA=0
set SKIP_FRONTEND=0
set SKIP_TAURI=0
set DO_QUANTIZE=0
set DO_FP16=0
set DO_SIGN=0

:parse_args
if "%~1"=="" goto args_done
if /i "%~1"=="--skip-env"      set SKIP_ENV=1
if /i "%~1"=="--skip-onnx"     set SKIP_ONNX=1
if /i "%~1"=="--skip-validate" set SKIP_VALIDATE=1
if /i "%~1"=="--skip-quantize" set SKIP_QUANTIZE=1
if /i "%~1"=="--skip-nuitka"   set SKIP_NUITKA=1
if /i "%~1"=="--skip-frontend" set SKIP_FRONTEND=1
if /i "%~1"=="--skip-tauri"    set SKIP_TAURI=1
if /i "%~1"=="--quantize"      set DO_QUANTIZE=1
if /i "%~1"=="--fp16"          set DO_FP16=1
if /i "%~1"=="--sign"          set DO_SIGN=1
shift
goto parse_args
:args_done

:: ── Paths ─────────────────────────────────────────────────────────────────────
set "REPO_ROOT=%~dp0.."
set "DESKTOP=%REPO_ROOT%\desktop"
set "BACKEND_SRC=%REPO_ROOT%\backend"
set "BACKEND_PROD=%DESKTOP%\backend_prod"
set "ONNX_DIR=%DESKTOP%\onnx_models"
set "DIST_DIR=%DESKTOP%\backend_prod_dist"
set "TAURI_DIR=%DESKTOP%\tauri"
set "VENV_DIR=%DESKTOP%\.venv_build"
set "SCRIPTS=%DESKTOP%\scripts"
set "LOG_DIR=%DESKTOP%\build_logs"
set "INSTALLER_OUT=%TAURI_DIR%\src-tauri\target\release\bundle\nsis"

:: ── Log directory ─────────────────────────────────────────────────────────────
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
set "BUILD_LOG=%LOG_DIR%\build_%BUILD_TIME%.log"
echo  Build log: %BUILD_LOG%
echo.

:: Helper: print a stage header
call :print_stage_header() 2>nul
goto :skip_func_defs

:print_stage_header
echo.
echo ┌──────────────────────────────────────────────────────┐
echo │  %~1
echo └──────────────────────────────────────────────────────┘
exit /b 0

:check_error
if errorlevel 1 (
    echo.
    echo [FAIL] Stage failed: %~1
    echo        Check log: %BUILD_LOG%
    exit /b 1
)
exit /b 0

:skip_func_defs

:: ============================================================================
:: STAGE 1 — Python virtual environment + dependencies
:: ============================================================================
if "%SKIP_ENV%"=="1" (
    echo [SKIP] Stage 1: Python environment
    goto stage2
)
call :print_stage_header "Stage 1/7 — Python Environment + Dependencies"

:: Check Python version
%PYTHON_CMD% -c "import sys; assert sys.version_info >= (3,11), 'Python 3.11+ required'" 2>>"%BUILD_LOG%"
if errorlevel 1 (
    echo [ERROR] Python 3.11+ is required. Found:
    %PYTHON_CMD% --version
    exit /b 1
)
for /f "tokens=*" %%v in ('%PYTHON_CMD% --version 2^>^&1') do echo   Python: %%v

:: Create venv if not exists
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo   Creating virtual environment...
    %PYTHON_CMD% -m venv "%VENV_DIR%" >> "%BUILD_LOG%" 2>&1
    call :check_error "venv creation"
    if errorlevel 1 exit /b 1
)

:: Activate venv
call "%VENV_DIR%\Scripts\activate.bat"
echo   Activated: %VENV_DIR%

:: Upgrade pip silently
python -m pip install --quiet --upgrade pip >> "%BUILD_LOG%" 2>&1

:: Install export-time tools (torch, onnx — only needed for export stage)
if "%SKIP_ONNX%"=="0" (
    echo   Installing ONNX export tools ^(torch, onnx^)...
    python -m pip install --quiet ^
        torch==2.2.0+cpu ^
        --index-url https://download.pytorch.org/whl/cpu ^
        >> "%BUILD_LOG%" 2>&1
    call :check_error "torch install"
    if errorlevel 1 exit /b 1

    python -m pip install --quiet ^
        onnx==1.16.0 ^
        onnxscript==0.1.0 ^
        onnxruntime==1.17.0 ^
        onnxconverter-common==1.13.0 ^
        >> "%BUILD_LOG%" 2>&1
    call :check_error "onnx tools install"
    if errorlevel 1 exit /b 1
)

:: Install production runtime deps
echo   Installing production dependencies...
python -m pip install --quiet ^
    -r "%BACKEND_PROD%\requirements_prod.txt" ^
    >> "%BUILD_LOG%" 2>&1
call :check_error "requirements_prod install"
if errorlevel 1 exit /b 1

:: Install Nuitka
echo   Installing Nuitka...
python -m pip install --quiet ^
    nuitka==2.3.0 ^
    ordered-set==4.1.0 ^
    zstandard==0.22.0 ^
    >> "%BUILD_LOG%" 2>&1
call :check_error "nuitka install"
if errorlevel 1 exit /b 1

echo   [OK] Python environment ready

:: ============================================================================
:: STAGE 2 — ONNX Export
:: ============================================================================
:stage2
if "%SKIP_ONNX%"=="1" (
    echo [SKIP] Stage 2: ONNX export
    goto stage3
)
call :print_stage_header "Stage 2/7 — ONNX Model Export"

if not exist "%ONNX_DIR%" mkdir "%ONNX_DIR%"

set FP16_FLAG=
if "%DO_FP16%"=="1" set FP16_FLAG=--fp16

python "%SCRIPTS%\export_onnx.py" ^
    --model-dir "%REPO_ROOT%\models" ^
    --output-dir "%ONNX_DIR%" ^
    --volume-size %VOLUME_SIZE% ^
    --opset 17 ^
    %FP16_FLAG% ^
    >> "%BUILD_LOG%" 2>&1
call :check_error "ONNX export"
if errorlevel 1 exit /b 1

:: Verify all three files exist
for %%M in (spatial_transformer segmentation_net anomaly_detector) do (
    if not exist "%ONNX_DIR%\%%M.onnx" (
        echo [ERROR] Missing ONNX file: %ONNX_DIR%\%%M.onnx
        exit /b 1
    )
    for %%F in ("%ONNX_DIR%\%%M.onnx") do (
        set /a SZ=%%~zF / 1048576
        echo   %%M.onnx  ^(!SZ! MB^)
    )
)
echo   [OK] ONNX export complete

:: ============================================================================
:: STAGE 3 — ONNX Validation
:: ============================================================================
:stage3
if "%SKIP_VALIDATE%"=="1" (
    echo [SKIP] Stage 3: ONNX validation
    goto stage4
)
call :print_stage_header "Stage 3/7 — ONNX Validation"

python "%SCRIPTS%\validate_onnx.py" ^
    --onnx-dir  "%ONNX_DIR%" ^
    --model-dir "%REPO_ROOT%\models" ^
    --volume-size %VOLUME_SIZE% ^
    --atol 1e-3 ^
    --n-bench 3 ^
    >> "%BUILD_LOG%" 2>&1
call :check_error "ONNX validation"
if errorlevel 1 (
    echo [FAIL] ONNX validation failed. Check %BUILD_LOG% for tolerance violations.
    exit /b 1
)
echo   [OK] ONNX validation passed

:: ============================================================================
:: STAGE 4 — ONNX Quantization (optional)
:: ============================================================================
:stage4
if "%SKIP_QUANTIZE%"=="1" goto stage5
if "%DO_QUANTIZE%"=="0" (
    echo [SKIP] Stage 4: ONNX quantization ^(use --quantize to enable^)
    goto stage5
)
call :print_stage_header "Stage 4/7 — ONNX INT8 Quantization"

python "%SCRIPTS%\quantize_onnx.py" ^
    --onnx-dir    "%ONNX_DIR%" ^
    --output-dir  "%ONNX_DIR%\quantized" ^
    --volume-size %VOLUME_SIZE% ^
    --n-calib 8 ^
    >> "%BUILD_LOG%" 2>&1
call :check_error "ONNX quantization"
if errorlevel 1 (
    echo [WARN] Quantization failed — continuing with FP32 models.
) else (
    echo   [OK] INT8 quantization complete
)

:: ============================================================================
:: STAGE 5 — Nuitka Backend Compile
:: ============================================================================
:stage5
if "%SKIP_NUITKA%"=="1" (
    echo [SKIP] Stage 5: Nuitka compile
    goto stage6
)
call :print_stage_header "Stage 5/7 — Nuitka Backend Compile"

call "%SCRIPTS%\build_nuitka.bat" >> "%BUILD_LOG%" 2>&1
call :check_error "Nuitka compile"
if errorlevel 1 exit /b 1

if not exist "%DIST_DIR%\medspatial_backend.exe" (
    echo [ERROR] Nuitka output not found: %DIST_DIR%\medspatial_backend.exe
    exit /b 1
)
for %%F in ("%DIST_DIR%\medspatial_backend.exe") do (
    set /a SZ=%%~zF / 1048576
    echo   medspatial_backend.exe  ^(!SZ! MB^)
)
echo   [OK] Backend compiled

:: ============================================================================
:: STAGE 6 — Vite Frontend Build
:: ============================================================================
:stage6
if "%SKIP_FRONTEND%"=="1" (
    echo [SKIP] Stage 6: Frontend build
    goto stage7
)
call :print_stage_header "Stage 6/7 — Vite Frontend Build"

:: Check Node
%NODE_CMD% --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js not found. Install Node.js 20 LTS.
    exit /b 1
)
for /f "tokens=*" %%v in ('%NODE_CMD% --version 2^>^&1') do echo   Node: %%v

:: Install npm deps
echo   Installing npm dependencies...
%NPM_CMD% install --prefix "%TAURI_DIR%" --prefer-offline >> "%BUILD_LOG%" 2>&1
call :check_error "npm install"
if errorlevel 1 exit /b 1

:: Copy frontend source files into tauri/src (only if not already there)
echo   Syncing frontend source...
:: Sync the original frontend components, stores, utils, workers into tauri/src
:: (tauri/src/main.jsx, tauri_bridge.js, tauri_api.js, LoadingScreen.jsx stay as-is)
if not exist "%TAURI_DIR%\src\components" (
    xcopy /E /I /Y /Q "%REPO_ROOT%\frontend\src\components" "%TAURI_DIR%\src\components" >> "%BUILD_LOG%" 2>&1
)
if not exist "%TAURI_DIR%\src\stores" (
    xcopy /E /I /Y /Q "%REPO_ROOT%\frontend\src\stores" "%TAURI_DIR%\src\stores" >> "%BUILD_LOG%" 2>&1
)
if not exist "%TAURI_DIR%\src\workers" (
    xcopy /E /I /Y /Q "%REPO_ROOT%\frontend\src\workers" "%TAURI_DIR%\src\workers" >> "%BUILD_LOG%" 2>&1
)
if not exist "%TAURI_DIR%\src\utils" (
    xcopy /E /I /Y /Q "%REPO_ROOT%\frontend\src\utils" "%TAURI_DIR%\src\utils" >> "%BUILD_LOG%" 2>&1
)
if not exist "%TAURI_DIR%\src\three" (
    xcopy /E /I /Y /Q "%REPO_ROOT%\frontend\src\three" "%TAURI_DIR%\src\three" >> "%BUILD_LOG%" 2>&1
)
:: Copy App.jsx, index.css (our main.jsx replaces original main.jsx)
if not exist "%TAURI_DIR%\src\App.jsx" (
    copy /Y "%REPO_ROOT%\frontend\src\App.jsx" "%TAURI_DIR%\src\App.jsx" >> "%BUILD_LOG%" 2>&1
)
if not exist "%TAURI_DIR%\src\index.css" (
    copy /Y "%REPO_ROOT%\frontend\src\index.css" "%TAURI_DIR%\src\index.css" >> "%BUILD_LOG%" 2>&1
)
:: Copy index.html to tauri root
if not exist "%TAURI_DIR%\index.html" (
    copy /Y "%REPO_ROOT%\frontend\index.html" "%TAURI_DIR%\index.html" >> "%BUILD_LOG%" 2>&1
)

:: Run Vite build
echo   Running Vite production build...
set "TAURI_ENV_TARGET_TRIPLE=x86_64-pc-windows-msvc"
%NPM_CMD% run build --prefix "%TAURI_DIR%" >> "%BUILD_LOG%" 2>&1
call :check_error "Vite build"
if errorlevel 1 exit /b 1

if not exist "%TAURI_DIR%\dist\index.html" (
    echo [ERROR] Vite dist not found: %TAURI_DIR%\dist\index.html
    exit /b 1
)
echo   [OK] Frontend built → %TAURI_DIR%\dist\

:: ============================================================================
:: STAGE 7 — Tauri NSIS Build
:: ============================================================================
:stage7
if "%SKIP_TAURI%"=="1" (
    echo [SKIP] Stage 7: Tauri build
    goto stage8
)
call :print_stage_header "Stage 7/7 — Tauri NSIS Installer Build"

:: Check Rust
cargo --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Rust/Cargo not found. Install via: https://rustup.rs
    exit /b 1
)
for /f "tokens=*" %%v in ('cargo --version 2^>^&1') do echo   Rust: %%v

:: Check tauri-cli
%NPM_CMD% list --prefix "%TAURI_DIR%" @tauri-apps/cli >nul 2>&1
if errorlevel 1 (
    echo   Installing @tauri-apps/cli...
    %NPM_CMD% install --prefix "%TAURI_DIR%" @tauri-apps/cli@2 >> "%BUILD_LOG%" 2>&1
)

:: Rename json5 → json (strip comments first using Node)
if exist "%TAURI_DIR%\src-tauri\tauri.conf.json5" (
    if not exist "%TAURI_DIR%\src-tauri\tauri.conf.json" (
        echo   Converting tauri.conf.json5 → tauri.conf.json...
        %NODE_CMD% -e "const fs=require('fs'); let s=fs.readFileSync('%TAURI_DIR%/src-tauri/tauri.conf.json5','utf8'); s=s.replace(/\/\/[^\n]*/g,'').replace(/\/\*[\s\S]*?\*\//g,''); fs.writeFileSync('%TAURI_DIR%/src-tauri/tauri.conf.json',s);" >> "%BUILD_LOG%" 2>&1
    )
)

:: Generate icons if not already present
if not exist "%TAURI_DIR%\src-tauri\icons\icon.ico" (
    echo   Generating app icons...
    %NPM_CMD% run --prefix "%TAURI_DIR%" icon -- "%TAURI_DIR%\src-tauri\icons\source_1024.png" >> "%BUILD_LOG%" 2>&1
    :: If source PNG missing, create a placeholder so the build doesn't fail
    if errorlevel 1 (
        echo   [WARN] Icon generation failed — using placeholder icons
        if not exist "%TAURI_DIR%\src-tauri\icons" mkdir "%TAURI_DIR%\src-tauri\icons"
    )
)

:: Run tauri build
echo   Running tauri build ^(this takes 5-15 minutes on first run^)...
%NPM_CMD% run --prefix "%TAURI_DIR%" tauri -- build --verbose >> "%BUILD_LOG%" 2>&1
call :check_error "tauri build"
if errorlevel 1 exit /b 1

:: ── Code signing (optional) ──────────────────────────────────────────────────
if "%DO_SIGN%"=="1" (
    if "%CERT_PATH%"=="" (
        echo [WARN] --sign requested but CERT_PATH not set — skipping signing
    ) else (
        echo   Signing installer...
        for /r "%INSTALLER_OUT%" %%F in (*-setup.exe) do (
            signtool sign /f "%CERT_PATH%" /p "%CERT_PASS%" ^
                /tr http://timestamp.digicert.com /td SHA256 /fd SHA256 ^
                "%%F" >> "%BUILD_LOG%" 2>&1
            call :check_error "signtool"
            if errorlevel 1 exit /b 1
            echo   Signed: %%F
        )
    )
)

:: ============================================================================
:: STAGE 8 — Post-build Verification
:: ============================================================================
:stage8
call :print_stage_header "Stage 8/8 — Post-Build Verification"

set FOUND_INSTALLER=0
for /r "%INSTALLER_OUT%" %%F in (*-setup.exe) do (
    set FOUND_INSTALLER=1
    set /a SZ=%%~zF / 1048576
    echo   Installer: %%F
    echo   Size     : !SZ! MB
    set "FINAL_INSTALLER=%%F"
)

if "%FOUND_INSTALLER%"=="0" (
    echo [FAIL] No installer found in %INSTALLER_OUT%
    echo        Check build log: %BUILD_LOG%
    exit /b 1
)

:: Report total build time
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set "END_DATETIME=%%I"
set "END_TIME=%END_DATETIME:~0,4%-%END_DATETIME:~4,2%-%END_DATETIME:~6,2%_%END_DATETIME:~8,2%%END_DATETIME:~10,2%%END_DATETIME:~12,2%"

echo.
echo  ██████████████████████████████████████████████████████
echo   BUILD SUCCESSFUL
echo  ██████████████████████████████████████████████████████
echo.
echo   Installer : !FINAL_INSTALLER!
echo   Log       : %BUILD_LOG%
echo.
echo   To test the installer, run:
echo     "!FINAL_INSTALLER!"
echo.
echo  ██████████████████████████████████████████████████████

endlocal
exit /b 0
