@echo off
REM ============================================================
REM  Lanzador del Separador Musical (usa SIEMPRE el venv local,
REM  que tiene torch con CUDA / GPU). No depende de activar nada.
REM ============================================================
setlocal
cd /d "%~dp0"

REM Verificar que el venv existe
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] No se encontro .venv\Scripts\python.exe
    echo         Crea el entorno primero:
    echo             py -3.12 -m venv .venv
    echo             .venv\Scripts\activate
    echo             pip install -r requirements.txt
    pause
    exit /b 1
)

echo [OK] Iniciando con el venv del proyecto...
".venv\Scripts\python.exe" main.py
pause