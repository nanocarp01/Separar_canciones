"""
Calidad de stems generados — política "no stems falsos".

Tras una separación, cada WAV se analiza para detectar:

  * Pistas **casi silenciosas** (pico por debajo de SILENT_PEAK_DB, -60 dBFS):
    señal de que el modelo no encontró material en esa banda. La pista se
    marca en el mezclador como ⚠ y se avisa; nunca se presenta como si
    contuviera audio real.
  * **Reconstrucción de la mezcla**: si los stems cubren el set completo
    (vocals + drums + bass + other), la suma debe acercarse al original.

Módulo independiente de la interfaz; usa solo soundfile + numpy.
"""

from __future__ import annotations

import math
from pathlib import Path

try:
    import numpy as np
except Exception:  # noqa: BLE001
    np = None  # type: ignore[assignment]
try:
    import soundfile as sf
except Exception:  # noqa: BLE001
    sf = None  # type: ignore[assignment]

# Por debajo de este pico (dBFS) una pista entera se considera "casi silenciosa".
SILENT_PEAK_DB = -60.0

# Set completo de stems: solo se puede verificar la reconstrucción si están todos.
FULL_STEM_SET = {"vocals", "drums", "bass", "other"}

_BLOCK = 1 << 20  # 1 Mframe por bloque de lectura


def _to_db(value: float) -> float:
    return 20.0 * math.log10(max(float(value), 1e-12))


def analyze_stem(path: str | Path) -> dict:
    """Analiza un WAV: metadatos, pico y RMS sin cargarlo entero en memoria.

    Devuelve un dict con ``silent`` (bool) y los valores medidos.
    """
    if sf is None or np is None:
        return {"path": str(path), "error": "Faltan soundfile/numpy.",
                "silent": False}
    p = Path(path)
    try:
        with sf.SoundFile(str(p)) as fh:
            sr = int(fh.samplerate)
            channels = int(fh.channels)
            frames = int(fh.frames)
            peak = 0.0
            sumsq = 0.0
            n = 0
            while True:
                data = fh.read(_BLOCK, dtype="float32")
                if data.size == 0:
                    break
                peak = max(peak, float(np.max(np.abs(data))))
                sumsq += float(np.sum(data.astype(np.float64) ** 2))
                n += int(data.size)
        rms = math.sqrt(sumsq / max(n, 1))
        peak_db = _to_db(peak)
        rms_db = _to_db(rms)
        return {
            "path": str(p),
            "name": p.name,
            "exists": True,
            "sr": sr,
            "channels": channels,
            "frames": frames,
            "duration_s": frames / max(sr, 1),
            "peak": peak,
            "peak_db": peak_db,
            "rms_db": rms_db,
            "silent": peak_db < SILENT_PEAK_DB,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "path": str(p), "name": p.name, "exists": False,
            "silent": False, "error": f"{type(exc).__name__}: {exc}",
        }


def validate_stems(paths: list[str | Path]) -> list[dict]:
    """Analiza todos los WAV de una separación y devuelve un informe por pista."""
    return [analyze_stem(p) for p in paths]


def is_silent(path: str | Path) -> bool:
    return bool(analyze_stem(path).get("silent"))


def check_reconstruction(original_path: str | Path,
                         stems: dict[str, str | Path]) -> dict | None:
    """Compara la suma de stems con el original.

    Solo es significativa si el set completo (vocals+drums+bass+other) está
    presente. Devuelve None si no aplica; si aplica, un dict con la diferencia
    RMS en dB (|diff| > 3 dB  ->  ``ok`` False).
    """
    if sf is None or np is None:
        return None
    if not FULL_STEM_SET.issubset(set(stems)):
        return None
    try:
        orig, sr = sf.read(str(original_path), always_2d=False, dtype="float32")
        if orig.ndim == 2:
            orig = orig.mean(axis=1)
        total = None
        for name in ("vocals", "drums", "bass", "other"):
            y, s = sf.read(str(stems[name]), always_2d=False, dtype="float32")
            if y.ndim == 2:
                y = y.mean(axis=1)
            if int(s) != int(sr):
                return {
                    "checked": False,
                    "note": (f"Frecuencias distintas: original {sr} Hz vs "
                             f"{name} {s} Hz; se omite la verificación."),
                    "ok": None,
                }
            total = y if total is None else total + y
        if total is None:
            return None
        n = min(len(orig), len(total))

        def rms_db(x: np.ndarray) -> float:
            r = float(np.sqrt(np.mean(x[:n].astype(np.float64) ** 2)))
            return _to_db(r)

        diff = rms_db(total) - rms_db(orig)
        return {
            "checked": True,
            "rms_orig_db": rms_db(orig),
            "rms_sum_db": rms_db(total),
            "diff_db": diff,
            "ok": abs(diff) <= 3.0,
            "note": f"suma de stems vs original: {diff:+.1f} dB",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "checked": False,
            "note": f"No se pudo verificar la reconstrucción: {exc}",
            "ok": None,
        }


def format_report(report: list[dict]) -> str:
    """Texto legible del informe de calidad de stems."""
    lines = ["🔊 Calidad de stems:"]
    for r in report:
        if r.get("error"):
            lines.append(f"  ✗ {r.get('name', '?')} — no se pudo leer: {r['error']}")
        elif r.get("silent"):
            lines.append(f"  ⚠ {r['name']} — CASI SILENCIOSA (pico {r['peak_db']:.0f} dBFS)")
        else:
            lines.append(f"  ✓ {r['name']} — pico {r['peak_db']:.0f} dBFS · "
                         f"RMS {r['rms_db']:.0f} dB · {r.get('duration_s', 0):.0f} s")
    return "\n".join(lines)
