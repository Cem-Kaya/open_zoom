@echo off
setlocal enabledelayedexpansion

set ROOT_DIR=%~dp0..

rem Map UNC paths (e.g. WSL shares) to a temporary drive letter so CUDA/MSBuild behave
pushd "%ROOT_DIR%" >nul
set ROOT_DIR=%CD%

set BUILD_DIR=%ROOT_DIR%\build
set QT_PREFIX_DEFAULT=C:\Qt\6.9.3\msvc2022_64

if not exist "%BUILD_DIR%" (
    mkdir "%BUILD_DIR%"
)

pushd "%BUILD_DIR%" >nul
set BUILD_DIR=%CD%

set CACHE_FILE=%BUILD_DIR%\CMakeCache.txt
if exist "%CACHE_FILE%" (
    set "BUILD_DIR_FWD=%BUILD_DIR:\=/%"
    findstr /I /C:"# For build in directory: !BUILD_DIR_FWD!" "%CACHE_FILE%" >nul
    if errorlevel 1 (
        echo Clearing stale CMake cache at %CACHE_FILE%
        del /f /q "%CACHE_FILE%" >nul 2>nul
        if exist "%BUILD_DIR%\CMakeFiles" (
            rmdir /s /q "%BUILD_DIR%\CMakeFiles"
        )
    )
)

if defined QT_PREFIX (
    set "CMAKE_QT_ARGS=-DCMAKE_PREFIX_PATH=%QT_PREFIX%"
) else if defined Qt6_DIR (
    set "CMAKE_QT_ARGS=-DQt6_DIR=%Qt6_DIR%"
) else if exist "%QT_PREFIX_DEFAULT%" (
    set "CMAKE_QT_ARGS=-DCMAKE_PREFIX_PATH=%QT_PREFIX_DEFAULT%"
) else (
    set "CMAKE_QT_ARGS="
)

cmake -S "%ROOT_DIR%" -B "%BUILD_DIR%" -G "Visual Studio 17 2022" -A x64 %CMAKE_QT_ARGS%
if errorlevel 1 goto :fail

cmake --build "%BUILD_DIR%" --config Release
if errorlevel 1 goto :fail

set EXE_PATH=%BUILD_DIR%\Release\open_zoom.exe
if exist "%EXE_PATH%" (
    echo Launching %EXE_PATH%
    "%EXE_PATH%"
) else (
    set EXE_PATH=%BUILD_DIR%\open_zoom.exe
    if exist "%EXE_PATH%" (
        echo Launching %EXE_PATH%
        "%EXE_PATH%"
    ) else (
        echo Executable not found. Build may have failed.
        goto :fail
    )
)

popd
popd
endlocal
exit /b 0

:fail
popd
popd
endlocal
exit /b 1
