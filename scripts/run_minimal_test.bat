@echo off
setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0
set ROOT_DIR=%SCRIPT_DIR%..

rem Ensure we operate on an absolute path (handles WSL shares/UNC)
pushd "%ROOT_DIR%" >nul
set ROOT_DIR=%CD%
set BUILD_DIR=%ROOT_DIR%\build
popd >nul

rem Build the main application without launching it.
set "OPENZOOM_SKIP_RUN=1"
call "%SCRIPT_DIR%build_and_run.bat"
if errorlevel 1 goto :fail
set "OPENZOOM_SKIP_RUN="

rem At this point the main app has exited and the build tree is configured.
if not exist "%BUILD_DIR%" (
    echo Build directory "%BUILD_DIR%" not found. Did the main build step succeed?
    goto :fail
)

set CONFIG=Release
set TARGET=dx12_cuda_minimal

rem Build the DX12/CUDA minimal validation target.
cmake --build "%BUILD_DIR%" --config %CONFIG% --target %TARGET%
if errorlevel 1 goto :fail

set EXE_PATH=%BUILD_DIR%\%CONFIG%\%TARGET%.exe
if not exist "%EXE_PATH%" set EXE_PATH=%BUILD_DIR%\sandbox\%TARGET%\%CONFIG%\%TARGET%.exe
if not exist "%EXE_PATH%" set EXE_PATH=%BUILD_DIR%\sandbox\%TARGET%\%TARGET%.exe
if not exist "%EXE_PATH%" set EXE_PATH=%BUILD_DIR%\%TARGET%.exe

if exist "%EXE_PATH%" (
    echo Running %TARGET% harness: "%EXE_PATH%"
    "%EXE_PATH%"
) else (
    echo Minimal harness executable not found at "%EXE_PATH%".
    goto :fail
)

endlocal
exit /b 0

:fail
endlocal
exit /b 1
