@echo off
REM ─────────────────────────────────────────────────────────────────────────────
REM run_qgis_render.bat
REM
REM Headless QGIS map renderer — VIIRS Electrification Analysis
REM
REM Sets up the full OSGeo4W runtime environment required for PyQGIS, then
REM runs scripts/qgis_render.py to produce 12 publication-quality maps
REM (9 individual A4 + 3 panel A3) at 300 DPI in PNG and PDF format.
REM
REM Output: figures\qgis_maps\
REM
REM Usage (from project root):
REM   scripts\run_qgis_render.bat
REM
REM Prerequisites:
REM   Run scripts\export_qgis_layers.py first to generate data\processed\*.gpkg
REM ─────────────────────────────────────────────────────────────────────────────

REM Move to the project root (one level above this script)
cd /d "%~dp0.."

REM ── OSGeo4W base environment ─────────────────────────────────────────────────
REM Loads OSGEO4W_ROOT, resets PATH to a minimal known-good state,
REM and sources all *.bat files in %OSGEO4W_ROOT%\etc\ini\
call "C:\Program Files\QGIS 3.40.11\bin\o4w_env.bat"
@echo off

REM ── Add QGIS and Qt binaries to PATH ─────────────────────────────────────────
REM Required for DLL resolution (qgis_core.dll, Qt5Core.dll, gdal*.dll, etc.)
set PATH=%OSGEO4W_ROOT%\apps\qgis-ltr\bin;%PATH%
set PATH=%OSGEO4W_ROOT%\apps\qt5\bin;%PATH%

REM ── QGIS / Qt environment variables ──────────────────────────────────────────
set QGIS_PREFIX_PATH=%OSGEO4W_ROOT:\=/%/apps/qgis-ltr
set QT_QPA_PLATFORM=offscreen
set QT_PLUGIN_PATH=%OSGEO4W_ROOT%\apps\qgis-ltr\qtplugins;%OSGEO4W_ROOT%\apps\qt5\plugins
set GDAL_FILENAME_IS_UTF8=YES
set VSI_CACHE=TRUE
set VSI_CACHE_SIZE=1000000

REM ── Python path — QGIS bindings ───────────────────────────────────────────────
set PYTHONPATH=%OSGEO4W_ROOT%\apps\qgis-ltr\python;%OSGEO4W_ROOT%\apps\qgis-ltr\python\plugins;%PYTHONPATH%

REM ── Run ───────────────────────────────────────────────────────────────────────
echo.
echo  VIIRS Electrification - Headless QGIS Map Renderer
echo  Project root: %CD%
echo.

python scripts\qgis_render.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo  ERROR: Renderer exited with code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo.
echo  All maps saved to figures\qgis_maps\
