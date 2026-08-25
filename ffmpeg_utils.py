"""
Utilidades de entorno: detección de FFmpeg y de GPU CUDA.

Reutilizado de la Versión 1 (main_desktop.py): audio-separator y demucs
invocan ``subprocess.run(["ffmpeg", "-version"])`` en cada separación. Si
ffmpeg no está en el PATH (lo habitual al instalarlo con winget sin reabrir
la terminal), reventaría. Estas funciones buscan el binario en ubicaciones
comunes y agregan el directorio resultante a ``os.environ["PATH"]``.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Optional

_FFMPEG_CANDIDATE_DIRS: list[str] = [
    # 1. PATH del sistema
    "",
    # 2. Winget portables (WindowsApps)
    os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WindowsApps"),
    # 3. Rutas comunes con binarios manual o portable
    r"C:\ffmpeg",
    r"C:\ffmpeg-local",
    r"C:\Program Files\ffmpeg",
    r"C:\Program Files (x86)\ffmpeg",
    r"C:\Tools\ffmpeg",
    # 4. Stow / unix-like
    "/usr/bin",
    "/usr/local/bin",
    "/opt/homebrew/bin",
    "/opt/local/bin",
]


def find_ffmpeg_dir() -> Optional[str]:
    """Devuelve el directorio donde está ffmpeg(.exe) o None si no lo encuentra."""
    found = shutil.which("ffmpeg")
    if found:
        return str(Path(found).parent)

    binary_name = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"

    for d in _FFMPEG_CANDIDATE_DIRS:
        if not d:
            continue
        candidate = Path(d) / binary_name
        if candidate.exists():
            return str(candidate.parent)

    # Búsqueda recursiva superficial (bin/ o ffmpeg-*-build), limitada por velocidad.
    for d in _FFMPEG_CANDIDATE_DIRS:
        if not d:
            continue
        base = Path(d)
        if not base.exists():
            continue
        try:
            for entry in base.iterdir():  # 1 nivel
                if entry.is_file() and entry.name.lower() == binary_name:
                    return str(entry.parent)
                if entry.is_dir() and entry.name.lower() == "bin":
                    inner = entry / binary_name
                    if inner.exists():
                        return str(inner.parent)
                if entry.is_dir() and "ffmpeg" in entry.name.lower():
                    inner = entry / "bin" / binary_name
                    if inner.exists():
                        return str(inner.parent)
                    inner2 = entry / binary_name
                    if inner2.exists():
                        return str(inner2.parent)
        except (PermissionError, OSError):
            continue
    return None


def ensure_ffmpeg_in_path() -> Optional[str]:
    """Asegura que ffmpeg esté en el PATH de este proceso. Devuelve la ruta."""
    ffmpeg_dir = find_ffmpeg_dir()
    if ffmpeg_dir is None:
        return None
    cur = os.environ.get("PATH", "")
    if ffmpeg_dir.lower() not in cur.lower().split(os.pathsep):
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + cur
    return ffmpeg_dir


# Al importar el módulo, dejamos ffmpeg listo para los motores.
ensure_ffmpeg_in_path()

# ---------------------------------------------------------------------------
# PyTorch / CUDA
# ---------------------------------------------------------------------------
try:
    import torch  # noqa: F401
    TORCH_AVAILABLE = True
    TORCH_IMPORT_ERROR = ""
except Exception as exc:  # noqa: BLE001
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False
    TORCH_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

DEVICE = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"


def cuda_available() -> bool:
    if not TORCH_AVAILABLE or torch is None:  # type: ignore[truthy-bool]
        return False
    try:
        return bool(
            torch.cuda.is_available()  # type: ignore[union-attr]
            and torch.cuda.device_count() > 0  # type: ignore[union-attr]
        )
    except Exception:  # noqa: BLE001
        return False


def gpu_name() -> Optional[str]:
    if not cuda_available() or torch is None:
        return None
    try:
        return str(torch.cuda.get_device_name(0))  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        return None
