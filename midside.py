"""
Filtro de cancelación de fase Mid/Side (Vocal Pan / Karaoke).

En temas en vivo el cantante principal suele estar paneado al CENTRO (aparece
igual en ambos canales), mientras que el público se capta con micrófonos
ambientales paneados a los lados (estéreo). La transformación Mid/Side
separa ambas cosas:

    Mid  = (L + R) / 2   → lo que está al centro (voz, bajo, bombo)
    Side = (L - R) / 2   → lo que está en los laterales (público, ambiente)

Presets disponibles (ver MIDSIDE_PRESETS):

  • "center"   → guarda solo el Mid  (quita el público estéreo; recomendado
                 para limpiar la pista de voz en directos).
  • "karaoke"  → guarda solo el Side (L−R: quita la voz central; clásico truco
                 karaoke para dejar la base).
  • "soft"     → Mid + 0.35·Side     (conserva la voz y atenúa el público).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

# (clave, etiqueta_UI, ayuda)
MIDSIDE_PRESETS: list[tuple[str, str, str]] = [
    (
        "center",
        "🎤 Conservar centro (M = L+R)",
        "Quita el sonido estéreo de los laterales (público/ambiente). "
        "Recomendado para limpiar la voz en temas en vivo.",
    ),
    (
        "karaoke",
        "🎤 Karaoke (L−R, quita la voz central)",
        "Elimina lo que está al centro (la voz) y deja los laterales. "
        "Sirve para obtener la base instrumental tipo karaoke.",
    ),
    (
        "soft",
        "🎚️ Atenuar público (M + 0.35·S)",
        "Conserva la voz casi intacta y baja los laterales (público) "
        "sin perder todo el ambiente.",
    ),
]

# Ganancia de centro y de lateral por preset
_PRESET_GAINS: dict[str, tuple[float, float]] = {
    "center": (1.0, 0.0),
    "karaoke": (0.0, 1.0),
    "soft": (1.0, 0.35),
}


def apply_midside_filter(
    input_path: str | Path,
    output_path: str | Path,
    mode: str = "center",
) -> Path:
    """Aplica cancelación de fase Mid/Side a un WAV estéreo.

    ``mode``: "center" | "karaoke" | "soft" (ver MIDSIDE_PRESETS).

    Sobrescribe o crea ``output_path`` con la mezcla resultante (estéreo,
    misma frecuencia de muestreo). Lanza ValueError si la pista es mono,
    porque la cancelación de fase necesita canales L/R.
    """
    if mode not in _PRESET_GAINS:
        raise ValueError(f"Modo Mid/Side desconocido: {mode!r}")

    data, sr = sf.read(str(input_path), always_2d=True, dtype="float32")
    if data.shape[1] < 2:
        raise ValueError(
            "La pista es MONO y la cancelación de fase necesita estéreo "
            "(canales L/R)."
        )

    left = data[:, 0]
    right = data[:, 1]
    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)

    cg, sg = _PRESET_GAINS[mode]
    out_left = cg * mid + sg * side
    out_right = cg * mid - sg * side
    out = np.stack([out_left, out_right], axis=1)

    # Evita clipping si la mezcla supera 1.0
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1.0:
        out = out / peak

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), out, sr)
    return out_path
