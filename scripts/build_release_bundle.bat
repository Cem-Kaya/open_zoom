@echo off
setlocal enabledelayedexpansion

rem Build and package OpenZoom into a self-contained dist/OpenZoom directory.
rem Requires a local Qt + (optionally) CUDA installation.

set ROOT_DIR=%~dp0..
pushd "%ROOT_DIR%" >nul
set ROOT_DIR=%CD%

set GENERATOR=%CMAKE_GENERATOR%
if not defined GENERATOR set GENERATOR=Visual Studio 17 2022

set CMAKE_ARCH_ARGS=
if /I "%GENERATOR%"=="Visual Studio 17 2022" set CMAKE_ARCH_ARGS=-A x64

set BUILD_DIR=%ROOT_DIR%\build\release-bundle
set DIST_DIR=%ROOT_DIR%\dist
set OUTPUT_DIR=%DIST_DIR%\OpenZoom
set QT_PREFIX_DEFAULT=C:\Qt\6.9.3\msvc2022_64
set CUDA_PREFIX_DEFAULT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0
set TESSERACT_PREFIX_DEFAULT=C:\Program Files\Tesseract-OCR

if not exist "%DIST_DIR%" mkdir "%DIST_DIR%"
if exist "%OUTPUT_DIR%" (
    echo Removing old bundle at %OUTPUT_DIR%
    rmdir /s /q "%OUTPUT_DIR%"
)
mkdir "%OUTPUT_DIR%"

rem -------------------------------------------------------------
rem Configure & build via plain CMake
rem -------------------------------------------------------------
set "CMAKE_QT_ARGS="
if defined QT_PREFIX (
    set "CMAKE_QT_ARGS=-DCMAKE_PREFIX_PATH=%QT_PREFIX%"
    call :resolve_qt_bindir "%QT_PREFIX%"
) else if defined Qt6_DIR (
    set "CMAKE_QT_ARGS=-DQt6_DIR=%Qt6_DIR%"
    call :resolve_qt_bindir "%Qt6_DIR%"
) else if exist "%QT_PREFIX_DEFAULT%" (
    set "QT_PREFIX=%QT_PREFIX_DEFAULT%"
    set "CMAKE_QT_ARGS=-DCMAKE_PREFIX_PATH=%QT_PREFIX_DEFAULT%"
    call :resolve_qt_bindir "%QT_PREFIX_DEFAULT%"
)

if not defined QT_BIN_DIR if exist "%QT_PREFIX_DEFAULT%" (
    set "QT_PREFIX=%QT_PREFIX_DEFAULT%"
    call :resolve_qt_bindir "%QT_PREFIX_DEFAULT%"
)

set "CMAKE_EXTRA_ARGS=%CMAKE_ARGS%"
if defined OPENZOOM_ENABLE_CUDA set "CMAKE_EXTRA_ARGS=%CMAKE_EXTRA_ARGS% -DOPENZOOM_ENABLE_CUDA=%OPENZOOM_ENABLE_CUDA%"

cmake -S "%ROOT_DIR%" -B "%BUILD_DIR%" -G "%GENERATOR%" %CMAKE_ARCH_ARGS% -DCMAKE_BUILD_TYPE=Release %CMAKE_QT_ARGS% %CMAKE_EXTRA_ARGS%
if errorlevel 1 goto :fail

if /I "%GENERATOR%"=="Visual Studio 17 2022" (
    cmake --build "%BUILD_DIR%" --config Release --target open_zoom
) else (
    cmake --build "%BUILD_DIR%" --target open_zoom
)
if errorlevel 1 goto :fail

set EXE_PATH=%BUILD_DIR%\cmake\Release\open_zoom.exe
if not exist "%EXE_PATH%" set EXE_PATH=%BUILD_DIR%\Release\open_zoom.exe
if not exist "%EXE_PATH%" set EXE_PATH=%BUILD_DIR%\open_zoom.exe
if not exist "%EXE_PATH%" (
    echo Built executable not found under %BUILD_DIR%.
    goto :fail
)

copy /y "%EXE_PATH%" "%OUTPUT_DIR%\open_zoom.exe" >nul
if errorlevel 1 goto :fail

rem -------------------------------------------------------------
rem Locate Qt runtime and deploy dependencies
rem -------------------------------------------------------------
set "QT_BIN_DIR="
set "WINDEPLOYQT="
if defined QT_PREFIX (
    call :resolve_qt_bindir "%QT_PREFIX%"
) else if defined Qt6_DIR (
    call :resolve_qt_bindir "%Qt6_DIR%"
) else if exist "%QT_PREFIX_DEFAULT%" (
    call :resolve_qt_bindir "%QT_PREFIX_DEFAULT%"
)

if not defined QT_BIN_DIR if exist "%QT_PREFIX_DEFAULT%" call :resolve_qt_bindir "%QT_PREFIX_DEFAULT%"

if defined QT_BIN_DIR if exist "%QT_BIN_DIR%\windeployqt.exe" (
    set "WINDEPLOYQT=%QT_BIN_DIR%\windeployqt.exe"
)

if not defined WINDEPLOYQT (
    if defined QT_PREFIX call :locate_windeployqt "%QT_PREFIX%"
)
if not defined WINDEPLOYQT (
    if defined Qt6_DIR call :locate_windeployqt "%Qt6_DIR%"
)
if not defined WINDEPLOYQT if exist "%QT_PREFIX_DEFAULT%" call :locate_windeployqt "%QT_PREFIX_DEFAULT%"

if defined WINDEPLOYQT (
    for %%D in ("%WINDEPLOYQT%") do set "QT_BIN_DIR=%%~dpD"
    echo Using Qt runtime from %QT_BIN_DIR%
    echo Running Qt deployment tool: %WINDEPLOYQT%
    "%WINDEPLOYQT%" --release "%OUTPUT_DIR%\open_zoom.exe" --dir "%OUTPUT_DIR%" --no-translations
) else (
    echo Warning: windeployqt.exe not found; Qt DLLs were not copied.
)

rem -------------------------------------------------------------
rem Copy CUDA redistributable runtime (if available)
rem -------------------------------------------------------------
call :copy_cuda_runtime "%OUTPUT_DIR%"

rem -------------------------------------------------------------
rem Copy optional local OCR runtime (if available)
rem -------------------------------------------------------------
call :copy_tesseract_runtime "%OUTPUT_DIR%"
if errorlevel 1 goto :fail

rem -------------------------------------------------------------
rem Copy ancillary files
rem -------------------------------------------------------------
if exist "%ROOT_DIR%\LICENSE" copy /y "%ROOT_DIR%\LICENSE" "%OUTPUT_DIR%\LICENSE" >nul
if exist "%ROOT_DIR%\README.md" copy /y "%ROOT_DIR%\README.md" "%OUTPUT_DIR%\README.txt" >nul
if exist "%ROOT_DIR%\docs\THIRD_PARTY_LICENSES.md" copy /y "%ROOT_DIR%\docs\THIRD_PARTY_LICENSES.md" "%OUTPUT_DIR%\THIRD_PARTY_LICENSES.md" >nul
if not exist "%OUTPUT_DIR%\licenses" mkdir "%OUTPUT_DIR%\licenses"
if exist "%ROOT_DIR%\assets\icons\lucide\LICENSE" copy /y "%ROOT_DIR%\assets\icons\lucide\LICENSE" "%OUTPUT_DIR%\licenses\LUCIDE_LICENSE.txt" >nul

popd

echo.
echo OpenZoom bundled build is ready at:
echo     %OUTPUT_DIR%
echo.
echo Contents:
dir /b "%OUTPUT_DIR%"
echo.
echo Launch from open_zoom.exe in that folder or zip the directory to share with others.
endlocal
exit /b 0

:resolve_qt_bindir
if defined QT_BIN_DIR goto :eof
set "QT_ROOT=%~1"
if not defined QT_ROOT goto :eof
for %%I in ("%QT_ROOT%") do set "QT_ROOT=%%~fI"
if exist "%QT_ROOT%\bin" (
    set "QT_BIN_DIR=%QT_ROOT%\bin"
    goto :eof
)
for %%I in ("%QT_ROOT%\..") do if not defined QT_BIN_DIR if exist "%%~fI\bin" set "QT_BIN_DIR=%%~fI\bin"
for %%I in ("%QT_ROOT%\..\..") do if not defined QT_BIN_DIR if exist "%%~fI\bin" set "QT_BIN_DIR=%%~fI\bin"
for %%I in ("%QT_ROOT%\..\..\..") do if not defined QT_BIN_DIR if exist "%%~fI\bin" set "QT_BIN_DIR=%%~fI\bin"
goto :eof

:locate_windeployqt
set "QT_SEARCH_ROOT=%~1"
if not defined QT_SEARCH_ROOT goto :eof
for %%I in ("%QT_SEARCH_ROOT%") do set "QT_SEARCH_ROOT=%%~fI"
for /f "delims=" %%Q in ('where /r "%QT_SEARCH_ROOT%" windeployqt.exe 2^>nul') do (
    set "WINDEPLOYQT=%%~fQ"
    goto :eof
)
goto :eof

:copy_tesseract_runtime
set "TESSERACT_BUNDLE_ROOT=%~1"
if not defined TESSERACT_BUNDLE_ROOT goto :eof
set "TESSERACT_SOURCE="

if defined OPENZOOM_TESSERACT_DIR if exist "%OPENZOOM_TESSERACT_DIR%\tesseract.exe" (
    set "TESSERACT_SOURCE=%OPENZOOM_TESSERACT_DIR%"
)
if not defined TESSERACT_SOURCE if defined TESSERACT_PREFIX if exist "%TESSERACT_PREFIX%\tesseract.exe" (
    set "TESSERACT_SOURCE=%TESSERACT_PREFIX%"
)
if not defined TESSERACT_SOURCE if exist "%TESSERACT_PREFIX_DEFAULT%\tesseract.exe" (
    set "TESSERACT_SOURCE=%TESSERACT_PREFIX_DEFAULT%"
)
if not defined TESSERACT_SOURCE (
    for /f "delims=" %%T in ('where tesseract.exe 2^>nul') do if not defined TESSERACT_SOURCE (
        for %%P in ("%%~fT") do set "TESSERACT_SOURCE=%%~dpP"
    )
)

if not defined TESSERACT_SOURCE (
    echo Tesseract OCR was not found; release bundle will not include local OCR.
    goto :eof
)

set "TESSERACT_DEST=%TESSERACT_BUNDLE_ROOT%\tools\tesseract"
if not exist "%TESSERACT_DEST%" mkdir "%TESSERACT_DEST%"
echo Bundling Tesseract OCR from %TESSERACT_SOURCE%

copy /y "%TESSERACT_SOURCE%\tesseract.exe" "%TESSERACT_DEST%\tesseract.exe" >nul
if errorlevel 1 goto :tesseract_failed
for %%F in ("%TESSERACT_SOURCE%\*.dll") do if exist "%%~fF" (
    copy /y "%%~fF" "%TESSERACT_DEST%" >nul
    if errorlevel 1 goto :tesseract_failed
)
if exist "%TESSERACT_SOURCE%\tessdata" (
    xcopy /e /i /y "%TESSERACT_SOURCE%\tessdata" "%TESSERACT_DEST%\tessdata" >nul
    if errorlevel 1 goto :tesseract_failed
)
if not exist "%TESSERACT_DEST%\licenses" mkdir "%TESSERACT_DEST%\licenses"
if exist "%TESSERACT_SOURCE%\doc\LICENSE" copy /y "%TESSERACT_SOURCE%\doc\LICENSE" "%TESSERACT_DEST%\licenses\TESSERACT_LICENSE.txt" >nul
if exist "%TESSERACT_SOURCE%\doc\AUTHORS" copy /y "%TESSERACT_SOURCE%\doc\AUTHORS" "%TESSERACT_DEST%\licenses\TESSERACT_AUTHORS.txt" >nul
if exist "%TESSERACT_SOURCE%\doc\README.md" copy /y "%TESSERACT_SOURCE%\doc\README.md" "%TESSERACT_DEST%\licenses\TESSERACT_README.md" >nul
echo Bundled Tesseract OCR runtime and language data.
goto :eof

:tesseract_failed
echo Failed to copy the Tesseract OCR runtime.
exit /b 1

:copy_cuda_runtime
set "DEST_DIR=%~1"
if not defined DEST_DIR goto :eof
set "CUDA_REDIST="
if defined CUDA_PATH (
    if exist "%CUDA_PATH%\redistributable_bin" (
        set "CUDA_REDIST=%CUDA_PATH%\redistributable_bin"
    ) else if exist "%CUDA_PATH%\bin\x64" (
        set "CUDA_REDIST=%CUDA_PATH%\bin\x64"
    ) else if exist "%CUDA_PATH%\bin" (
        set "CUDA_REDIST=%CUDA_PATH%\bin"
    )
)
if not defined CUDA_PATH if exist "%CUDA_PREFIX_DEFAULT%" set "CUDA_PATH=%CUDA_PREFIX_DEFAULT%"
if not defined CUDA_REDIST if exist "%CUDA_PREFIX_DEFAULT%\bin\x64" set "CUDA_REDIST=%CUDA_PREFIX_DEFAULT%\bin\x64"
if not defined CUDA_REDIST (
    for /f "tokens=1* delims==" %%A in ('set CUDA_PATH_V 2^>nul') do (
        if not defined CUDA_REDIST (
            for %%P in ("%%B") do (
                if exist "%%~fP\redistributable_bin" (
                    set "CUDA_REDIST=%%~fP\redistributable_bin"
                ) else if exist "%%~fP\bin\x64" (
                    set "CUDA_REDIST=%%~fP\bin\x64"
                ) else if exist "%%~fP\bin" (
                    set "CUDA_REDIST=%%~fP\bin"
                )
            )
        )
    )
)
if not defined CUDA_REDIST goto :cuda_done

echo Using CUDA redistributables from %CUDA_REDIST%

set COPIED=0
set "CUDA_DLL_LIST=cudart64_13.dll nvrtc64_130_0.dll nvrtc64_130_0.alt.dll nvrtc-builtins64_130.dll nvJitLink_130_0.dll nvfatbin_130_0.dll"
for %%F in (%CUDA_DLL_LIST%) do (
    if exist "%CUDA_REDIST%\%%F" (
        copy /y "%CUDA_REDIST%\%%F" "%DEST_DIR%" >nul
        set COPIED=1
    )
)
for %%F in (cudart64*.dll nvrtc64*.dll) do (
    for %%P in ("%CUDA_REDIST%\%%F") do (
        if exist %%~fP (
            copy /y %%~fP "%DEST_DIR%" >nul
            set COPIED=1
        )
    )
)
if !COPIED! EQU 0 (
    echo Warning: No CUDA redistributable DLLs copied from %CUDA_REDIST%.
) else (
    echo Copied CUDA redistributable DLLs from %CUDA_REDIST%.
)
:cuda_done
goto :eof

:fail
popd
endlocal
exit /b 1
