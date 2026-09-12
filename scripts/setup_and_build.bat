@echo off
REM Windows launcher for setup_and_build.ps1
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_and_build.ps1" %*
exit /b %ERRORLEVEL%
