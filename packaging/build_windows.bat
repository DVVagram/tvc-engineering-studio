@echo off
rem Build the Windows onedir distribution of TVC Engineering Studio.
setlocal
cd /d "%~dp0\.."

if not exist ".venv\Scripts\python.exe" (
    echo [build] creating virtual environment
    py -3.13 -m venv .venv || goto :error
)

echo [build] installing runtime + packaging dependencies
".venv\Scripts\python.exe" -m pip install --upgrade pip || goto :error
".venv\Scripts\python.exe" -m pip install -e ".[packaging,hardware]" || goto :error

echo [build] running headless self-test
".venv\Scripts\python.exe" -m tvc_studio --self-test || goto :error

echo [build] running validation suite
".venv\Scripts\python.exe" -m pytest || goto :error

echo [build] packaging with PyInstaller
".venv\Scripts\python.exe" -m PyInstaller --noconfirm --clean packaging\tvc_studio.spec || goto :error

echo.
echo [build] done: dist\TVC Engineering Studio\TVC Engineering Studio.exe
echo [build] optional installer: compile packaging\installer.iss with Inno Setup 6
exit /b 0

:error
echo.
echo [build] FAILED with exit code %errorlevel%
exit /b %errorlevel%
