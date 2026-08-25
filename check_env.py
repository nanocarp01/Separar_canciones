#!/usr/bin/env python3
"""
check_env.py — Diagnóstico de dependencias para Demucs + RoFormer + Tkinter.

Ejecuta este script con el MISMO Python que usarás para `main.py`:

    .venv/Scripts/python.exe check_env.py     # Windows (venv)
    python check_env.py                       # sistema

y compara la salida con el comando que usaste para instalar paquetes.

Uso típico para corregir problemas:

    # SIEMPRE usa `python -m pip ...` para apuntar al mismo intérprete
    python -m pip install -r requirements.txt
"""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
from pathlib import Path

# En Windows la consola va por cp1252 y rompe con caracteres no ASCII.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass


WIDTH = 64


def line(char: str = "=") -> None:
    print(char * WIDTH)


def safe_import(name: str):
    try:
        return importlib.import_module(name), ""
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def main() -> int:
    line("#")
    print("#  Diagnóstico de entorno · Demucs + RoFormer + Tkinter")
    line("#")
    print(f"Python ejecutable : {sys.executable}")
    print(f"Versión           : {sys.version.split()[0]}")
    print(f"Plataforma        : {sys.platform}")
    print(f"Virtualenv        : {sys.prefix != sys.base_prefix}"
          f"  (ruta: {sys.prefix})")
    print()
    print("Primero, averigua si tu `python` coincide con el `pip` que usaste:")
    print("  pip --version             muestra la ruta del Python al que apunta")
    print("  python -m pip --version   SIEMPRE apunta al mismo python")
    line()

    # --- torch ----------------------------------------------------------
    torch, err = safe_import("torch")
    if torch is not None:
        print(f"OK torch                {torch.__version__}")
        print(f"                         {torch.__file__}")
        try:
            cuda = bool(torch.cuda.is_available())
            print(f"                         CUDA disponible: {'si' if cuda else 'no'}")
            if cuda:
                try:
                    print(f"                         GPU: {torch.cuda.get_device_name(0)}")
                    print(f"                         Capability: {torch.cuda.get_device_capability(0)}")
                except Exception as exc:  # noqa: BLE001
                    print(f"                         (no pude leer nombre: {exc})")
        except Exception as exc:  # noqa: BLE001
            print(f"                         Error CUDA: {exc}")
    else:
        print(f"NO torch NO instalado   ->  {err}")
    print()

    # --- torchaudio -----------------------------------------------------
    ta, err = safe_import("torchaudio")
    if ta is not None:
        print(f"OK torchaudio           {ta.__version__}")
        print(f"                         {ta.__file__}")
    else:
        print(f"NO torchaudio NO instalado ->  {err}")
    print()

    # --- demucs --------------------------------------------------------
    dm, err = safe_import("demucs")
    if dm is None:
        print("NO demucs NO esta instalado en este Python.")
        print("   Instalacion sugerida: python -m pip install demucs")
    else:
        print(f"OK demucs               {getattr(dm, '__version__', '?')}")
        print(f"                         {dm.__file__}")
        api, err2 = safe_import("demucs.api")
        if api is not None and hasattr(api, "Separator"):
            print("OK demucs.api.Separator disponible (Demucs 4.x API)")
        else:
            print(f"NO demucs.api.Separator NO disponible -> {err2}")
            print("   Tu version de Demucs es muy antigua. Actualiza con: python -m pip install -U demucs")
    print()

    # --- audio-separator (motor RoFormer) ----------------------------
    asep, err = safe_import("audio_separator")
    if asep is not None:
        try:
            from importlib.metadata import version as _pkg_version
            asep_version = _pkg_version("audio-separator")
        except Exception:  # noqa: BLE001
            asep_version = "(version desconocida)"
        print(f"OK audio-separator      {asep_version}")
        print(f"                         {asep.__file__ if hasattr(asep, '__file__') else '(paquete)'}")
        apis, err2 = safe_import("audio_separator.separator")
        if apis is not None and hasattr(apis, "Separator"):
            print("OK audio-separator.Separator disponible (RoFormer, MDXC, Demucs)")
        else:
            print(f"NO audio-separator.Separator NO disponible -> {err2}")
    else:
        print(f"NO audio-separator NO instalado ->  {err}")
        print("   Motor RoFormer no disponible; Demucs seguira funcionando.")
    print()

    # --- ffmpeg (binario del sistema) -------------------------------
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        try:
            out = subprocess.check_output(
                ["ffmpeg", "-version"], text=True, timeout=10
            ).splitlines()
            ver = out[0] if out else "?"
            print(f"OK ffmpeg     {ffmpeg_path}")
            print(f"               {ver}")
        except Exception as exc:  # noqa: BLE001
            print(f"NO ffmpeg     {ffmpeg_path}  (no responde: {exc})")
    else:
        # Búsqueda exhaustiva (la misma que usa main.py): conocido C:\ffmpeg-local, etc.
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from main import _find_ffmpeg_dir as _ffd  # type: ignore
            auto_dir = _ffd()
        except Exception:  # noqa: BLE001
            auto_dir = None

        if auto_dir:
            print(f"-- ffmpeg     NO esta en PATH, pero main.py lo encontro en:")
            print(f"               {auto_dir}")
            print(f"               La GUI funcionara correctamente.")
        else:
            print("NO ffmpeg     no encontrado en el PATH del sistema")
            print("               (ni en C:\\ffmpeg, C:\\ffmpeg-local, etc.)")
            print("               Necesario para audio-separator (decodificar MP3, etc.).")
            print("               Instalacion:")
            print("                 Windows:  winget install -e --id Gyan.FFmpeg")
            print("                 macOS:    brew install ffmpeg")
            print("                 Linux:    sudo apt-get install ffmpeg")
    print()

    # --- onnxruntime providers (GPU) --------------------------------
    ort, err = safe_import("onnxruntime")
    if ort is not None:
        try:
            providers = ort.get_available_providers()
            print(f"OK onnxruntime  {ort.__version__}  providers={providers}")
            if "CUDAExecutionProvider" in providers:
                print("               -> aceleracion GPU activa (CUDAExecutionProvider)")
            else:
                print("               -> solo CPU; onnxruntime-gpu no instalado")
        except Exception:  # noqa: BLE001
            print(f"OK onnxruntime  {ort.__version__}")
    else:
        print(f"NO onnxruntime NO instalado ->  {err}")
    print()

    line()
    print("Resumen:")
    print(f"  torch               : {'OK' if torch else 'FALTA'}")
    print(f"  torchaudio          : {'OK' if ta else 'FALTA'}")
    api_ok = False
    if dm:
        a, _ = safe_import("demucs.api")
        api_ok = a is not None and hasattr(a, "Separator")
    print(f"  demucs 4.x          : {'OK' if api_ok else 'FALTA'}")
    print(f"  audio-separator     : {'OK' if asep else 'FALTA'}")
    print(f"  ffmpeg en PATH      : {'OK' if ffmpeg_path else 'FALTA'}")
    cuda_ok = bool(torch and torch.cuda.is_available())
    print(f"  CUDA detectada      : {'OK' if cuda_ok else 'NO'}")
    line()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
