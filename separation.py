"""
Separación de fuentes con Demucs y mezclador multipista.

Usa la misma API que la Versión 1 (``demucs.api.Separator``), con detección
automática de GPU CUDA (``ffmpeg_utils.DEVICE``) y respaldo a CPU.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import soundfile as sf

from ffmpeg_utils import DEVICE

logger = logging.getLogger("Separacion")

# Nombres internos -> etiquetas en español
STEM_INFO = {
    "vocals": {"es": "🎤 Voz"},
    "drums":  {"es": "🥁 Batería"},
    "bass":   {"es": "🎸 Bajo"},
    "other":  {"es": "🎧 Otros"},
    "guitar": {"es": "🎸 Guitarra"},
    "piano":  {"es": "🎹 Piano"},
}
STEM_ORDER = ["vocals", "drums", "bass", "other", "guitar", "piano"]

# Modos de separación: etiqueta -> (modelo Demucs, stems que produce)
MODE_CHOICES = [
    ("4 Pistas (Voz, Batería, Bajo, Otros)", "htdemucs"),
    ("6 Pistas (Voz, Batería, Bajo, Guitarra, Piano, Otros)", "htdemucs_6s"),
]
MODEL_STEMS = {
    "htdemucs":    ["vocals", "drums", "bass", "other"],
    "htdemucs_6s": ["vocals", "drums", "bass", "other", "guitar", "piano"],
}


def separate(audio_path: str, model_name: str, out_dir: str) -> dict[str, str]:
    """Separa ``audio_path`` con el modelo Demucs y guarda cada stem como WAV.

    Devuelve ``{nombre_stem: ruta_wav}``. Lanza ``RuntimeError`` con un
    mensaje legible si algo falla.
    """
    try:
        from demucs.api import Separator

        # Menos shifts en CPU para no hacer la espera eterna
        shifts = 1 if DEVICE == "cuda" else 0
        logger.info("Cargando modelo '%s' en %s (la 1ª vez se descarga)...",
                    model_name, DEVICE)
        separator = Separator(
            model=model_name,
            device=DEVICE,
            shifts=shifts,
            split=True,
            overlap=0.25,
            progress=False,
        )

        logger.info("Separando %s ...", audio_path)
        _, separated = separator.separate_audio_file(audio_path)
        sr = int(getattr(separator, "samplerate", 44100))

        stems: dict[str, str] = {}
        for idx, name in enumerate(MODEL_STEMS[model_name]):
            w = separated[name].detach().cpu()
            if w.ndim == 3:
                w = w[0]
            data = w.numpy().T.astype("float32", copy=False)  # (samples, canales)
            out_path = os.path.join(out_dir, f"{idx + 1:02d}_{name}.wav")
            sf.write(out_path, data, sr)
            stems[name] = out_path
        logger.info("Separación completada: %s", list(stems))
        return stems
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error separando con %s", model_name)
        raise RuntimeError(f"Error separando con '{model_name}': {exc}") from exc


def mix_stems(stems: dict[str, str], gains: dict[str, float], out_path: str) -> str:
    """Mezcla los stems con los volúmenes (0-200 %) y escribe el WAV resultante.

    Devuelve la ruta del archivo de mezcla. ``gains`` usa como claves los
    nombres internos de los stems; un valor ausente se interpreta como 100 %.
    """
    if not stems:
        raise RuntimeError("No hay stems para mezclar.")

    mixed, sr = None, None
    for name, path in stems.items():
        y, s = sf.read(path, always_2d=True)  # (frames, canales)
        g = max(0.0, gains.get(name, 100.0)) / 100.0
        mixed = y * g if mixed is None else mixed + y * g
        sr = s

    mixed = np.clip(mixed, -1.0, 1.0)  # evita saturación
    sf.write(out_path, mixed, sr)
    return out_path
