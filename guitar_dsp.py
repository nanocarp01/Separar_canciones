"""Separación de guitarras por DSP clásico (sin modelos de machine learning).

Dos métodos independientes, pensados para aplicarse sobre ``guitar.wav``:

* ``separate_guitars_by_panning`` — inspirado en ADRess (Azimuth
  Discrimination and Resynthesis, Barry et al. 2004). Calcula la posición
  estéreo de cada celda tiempo-frecuencia, detecta las posiciones dominantes
  (p. ej. guitarra doblada a la izquierda y a la derecha) y reconstruye cada
  una con máscaras suaves. Funciona bien cuando las guitarras están paneadas a
  lados distintos; en pistas mono no puede separar nada.

* ``separate_lead_by_melody`` — sigue la melodía predominante por suma de
  armónicos (la lead suele ser monofónica), construye una máscara armónica alrededor de
  f0 y sus armónicos y genera ``guitar_lead.wav``; el residuo es
  ``guitar_rhythm.wav``. Si la rítmica toca las mismas notas que la lead,
  parte de ella se filtrará en la lead (limitación inherente del método).

Ambos conservan la imagen estéreo: la máscara se aplica a cada canal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import librosa
import numpy as np
import soundfile as sf
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, uniform_filter1d
from scipy.signal import find_peaks, medfilt

Progress = Callable[[str], None] | None


# --------------------------------------------------------------- utilidades
def _report(progress: Progress, text: str) -> None:
    if progress:
        progress(text)


def _load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Devuelve audio (canales, muestras) float32 y sample rate original."""
    source = Path(path)
    if not source.exists() or source.stat().st_size == 0:
        raise ValueError("El archivo de guitarra no existe o está vacío.")
    audio, sr = librosa.load(str(source), sr=None, mono=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[None, :]
    if not np.isfinite(audio).all() or np.max(np.abs(audio)) < 1e-8:
        raise ValueError("La pista de guitarra no contiene señal audible.")
    return audio, int(sr)


def _write(path: Path, channels: list[np.ndarray], sr: int) -> Path:
    result = np.stack(channels, axis=1).astype(np.float32)
    peak = float(np.max(np.abs(result))) if result.size else 0.0
    if peak > 0.99:
        result *= 0.98 / peak
    sf.write(str(path), result, sr)
    return path


def _resynth(stfts: list[np.ndarray], mask: np.ndarray, hop: int, length: int) -> list[np.ndarray]:
    out = []
    for stft in stfts:
        signal = librosa.istft(stft * mask, hop_length=hop, length=length)
        out.append(np.asarray(signal, dtype=np.float32)[:length])
    return out


# ------------------------------------------------- 1) separación por paneo
def separate_guitars_by_panning(
    input_path: str | Path,
    output_dir: str | Path,
    max_sources: int = 3,
    width: float = 0.18,
    n_fft: int = 4096,
    hop_length: int = 1024,
    progress: Progress = None,
) -> list[Path]:
    """Separa guitarras según su posición en el panorama estéreo.

    Genera ``guitar_pan_izq.wav``, ``guitar_pan_der.wav`` y/o
    ``guitar_pan_centro.wav`` según las posiciones que encuentre.
    ``width`` controla qué tan estricta es cada máscara (0.1 = muy estricta,
    0.3 = más suave, con más mezcla entre fuentes).
    """
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    _report(progress, "Leyendo guitarra estéreo…")
    audio, sr = _load_audio(input_path)
    if audio.shape[0] < 2:
        raise ValueError("La pista es mono: la separación por paneo necesita audio estéreo.")
    left, right = audio[0], audio[1]
    length = audio.shape[1]

    _report(progress, "Calculando espectrogramas L/R…")
    stft_l = librosa.stft(left, n_fft=n_fft, hop_length=hop_length)
    stft_r = librosa.stft(right, n_fft=n_fft, hop_length=hop_length)
    mag_l, mag_r = np.abs(stft_l), np.abs(stft_r)
    total = mag_l + mag_r + 1e-10

    # Índice de paneo por celda: -1 = todo izquierda, 0 = centro, +1 = derecha.
    pan = ((mag_r - mag_l) / total).astype(np.float32)
    energy = (mag_l ** 2 + mag_r ** 2).astype(np.float32)

    _report(progress, "Buscando posiciones estéreo dominantes…")
    bins = 101
    hist, edges = np.histogram(pan, bins=bins, range=(-1.0, 1.0), weights=energy)
    hist = gaussian_filter1d(hist.astype(np.float64), sigma=2.0)
    if hist.max() <= 0:
        raise ValueError("No se encontró energía suficiente para analizar el paneo.")
    hist /= hist.max()
    centers_axis = (edges[:-1] + edges[1:]) / 2
    peaks, props = find_peaks(hist, height=0.08, distance=int(bins * 0.12), prominence=0.05)
    if len(peaks) == 0:
        raise ValueError("No se detectaron posiciones estéreo claras.")
    order = np.argsort(props["peak_heights"])[::-1][:max_sources]
    centers = sorted(float(centers_axis[peaks[i]]) for i in order)
    if len(centers) < 2:
        raise ValueError(
            f"Solo se detectó una posición estéreo (≈{centers[0]:+.2f}). Las guitarras "
            "parecen estar en el mismo lugar del panorama; prueba el método por melodía.")

    _report(progress, f"Posiciones encontradas: {', '.join(f'{c:+.2f}' for c in centers)}. Creando máscaras…")
    weights = np.stack([np.exp(-((pan - c) ** 2) / (2 * width ** 2)) for c in centers])
    masks = weights / (weights.sum(axis=0, keepdims=True) + 1e-8)

    outputs: list[Path] = []
    used: set[str] = set()
    for index, center in enumerate(centers):
        if center < -0.15:
            name = "izq"
        elif center > 0.15:
            name = "der"
        else:
            name = "centro"
        if name in used:
            name = f"{name}_{index + 1}"
        used.add(name)
        _report(progress, f"Reconstruyendo guitar_pan_{name}.wav…")
        channels = _resynth([stft_l, stft_r], masks[index], hop_length, length)
        outputs.append(_write(destination / f"guitar_pan_{name}.wav", channels, sr))
    return outputs


# ---------------------------------------- 2) lead por seguimiento de melodía
# Umbral de "energía explicada por la nota seguida" (promedio de 4 s) para
# considerar que una sección tiene lead. Calibrado con Lamento Boliviano:
# rasgueo 0.15–0.29, solo 0.38–0.58, arreglos de lead sobre rítmica 0.25–0.36.
SENSITIVITY_THRESHOLDS = {"baja": 0.42, "media": 0.34, "alta": 0.27}


def _harmonic_mask(f0: np.ndarray, freqs: np.ndarray, sr: int, n_fft: int,
                   n_harmonics: int, width_cents: float) -> np.ndarray:
    """Máscara (frecuencias, frames) con gaussianas en f0 y sus armónicos."""
    bin_hz = float(sr) / n_fft
    nyquist = sr / 2
    mask = np.zeros((freqs.shape[0], len(f0)), dtype=np.float32)
    f0_row = np.nan_to_num(f0, nan=0.0).astype(np.float32)[None, :]
    cents_factor = 2.0 ** (width_cents / 1200.0) - 1.0
    for h in range(1, n_harmonics + 1):
        centers = h * f0_row
        active = (centers > 0) & (centers < nyquist)
        if not active.any():
            break
        sigma = np.maximum(centers * cents_factor, bin_hz)
        contribution = np.exp(-((freqs - centers) ** 2) / (2 * sigma ** 2))
        contribution *= active
        np.maximum(mask, contribution, out=mask)
    return mask


def _clean_runs(active: np.ndarray, min_on: int, max_gap: int) -> np.ndarray:
    """Une huecos cortos entre tramos activos y descarta tramos muy cortos."""
    active = active.copy()
    n = len(active)
    # Rellenar huecos cortos entre dos tramos activos.
    i = 0
    while i < n:
        if not active[i]:
            j = i
            while j < n and not active[j]:
                j += 1
            if 0 < i and j < n and (j - i) <= max_gap:
                active[i:j] = True
            i = j
        else:
            i += 1
    # Quitar tramos más cortos que min_on.
    i = 0
    while i < n:
        if active[i]:
            j = i
            while j < n and active[j]:
                j += 1
            if (j - i) < min_on:
                active[i:j] = False
            i = j
        else:
            i += 1
    return active


def _format_runs(active: np.ndarray, fps: float) -> str:
    def mmss(sec: float) -> str:
        sec = int(round(sec))
        return f"{sec // 60}:{sec % 60:02d}"

    runs, i, n = [], 0, len(active)
    while i < n:
        if active[i]:
            j = i
            while j < n and active[j]:
                j += 1
            runs.append(f"{mmss(i / fps)}–{mmss(j / fps)}")
            i = j
        else:
            i += 1
    return ", ".join(runs) if runs else "ninguna"


def separate_lead_by_melody(
    input_path: str | Path,
    output_dir: str | Path,
    fmin_note: str = "D3",
    fmax_note: str = "E6",
    n_harmonics: int = 20,
    width_cents: float = 40.0,
    extract_width_cents: float = 80.0,
    normalize_lead: bool = True,
    strong_ratio: float = 0.7,
    voicing_threshold: float = 0.25,
    sensitivity: str = "media",
    section_seconds: float = 4.0,
    min_section_seconds: float = 3.0,
    max_gap_seconds: float = 2.0,
    frame_threshold: float = 0.15,
    n_fft: int = 4096,
    hop_length: int = 1024,
    progress: Progress = None,
) -> list[Path]:
    """Extrae la guitarra lead (monofónica) y deja el resto como rítmica.

    Genera ``guitar_lead.wav`` y ``guitar_rhythm.wav``.

    La melodía se estima por *suma de armónicos* (estilo Melodia): para cada
    f0 candidata se suma la energía en sus armónicos. Entre los candidatos con
    saliencia ≥ ``strong_ratio`` del máximo se elige el más agudo, porque la
    lead suele sonar por encima de los acordes (evita que gane la nota grave de
    un acorde cuyos armónicos coinciden con la lead).
    """
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    _report(progress, "Leyendo guitarra…")
    audio, sr = _load_audio(input_path)
    length = audio.shape[1]
    mono = np.mean(audio, axis=0)

    _report(progress, "Calculando espectrograma…")
    stfts = [librosa.stft(ch, n_fft=n_fft, hop_length=hop_length) for ch in audio]
    freq_bins, frames = stfts[0].shape
    bin_hz = float(sr) / n_fft

    _report(progress, "Estimando la melodía predominante (suma de armónicos)…")
    spec_raw = np.abs(librosa.stft(mono, n_fft=n_fft, hop_length=hop_length)).astype(np.float32)
    spec = np.log1p(100.0 * spec_raw / (spec_raw.max() + 1e-10))  # compresión: equilibra graves/agudos
    fmin = librosa.note_to_hz(fmin_note)
    fmax = librosa.note_to_hz(fmax_note)
    step_cents = 10
    grid = fmin * 2.0 ** (np.arange(0, 1200 * np.log2(fmax / fmin), step_cents) / 1200.0)
    salience = np.zeros((len(grid), frames), dtype=np.float32)
    for h in range(1, 9):
        idx = h * grid / bin_hz
        ok = idx < freq_bins - 1
        lo = np.floor(idx[ok]).astype(int)
        frac = (idx[ok] - lo)[:, None].astype(np.float32)
        salience[ok] += (0.8 ** (h - 1)) * (spec[lo] * (1 - frac) + spec[lo + 1] * frac)

    peak_sal = salience.max(axis=0)
    strong = salience >= strong_ratio * peak_sal[None, :]
    highest = len(grid) - 1 - np.argmax(strong[::-1], axis=0)
    # Afinar: máximo real dentro de ±50 cents del candidato elegido.
    half = 50 // step_cents
    offsets = np.arange(-half, half + 1)
    window = np.clip(highest[None, :] + offsets[:, None], 0, len(grid) - 1)
    refined = window[np.argmax(salience[window, np.arange(frames)[None, :]], axis=0), np.arange(frames)]
    # Para extraer se usa un suavizado corto (sigue bends y notas rápidas);
    # para detectar secciones, el suavizado largo original (calibrado).
    f0_extract = grid[medfilt(refined.astype(np.float64), 3).astype(int)]
    refined = medfilt(refined.astype(np.float64), 7).astype(int)
    f0_frames = grid[refined]

    # Sonoridad: frames donde la melodía tiene saliencia suficiente.
    path_sal = salience[refined, np.arange(frames)]
    reference = np.percentile(path_sal, 95) + 1e-10
    voiced = path_sal >= voicing_threshold * reference
    if np.count_nonzero(voiced) < 10:
        raise ValueError("No se detectó una melodía clara: puede que no haya guitarra lead en esta pista.")
    f0_frames = np.where(voiced, f0_frames, np.nan)
    f0_extract = np.where(voiced, f0_extract, np.nan)
    gain = medfilt(voiced.astype(np.float64), 5).astype(np.float32)

    _report(progress, f"Construyendo máscara armónica ({n_harmonics} armónicos)…")
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft).astype(np.float32)[:, None]
    # Máscara de detección: colador angosto (40 cents) con el que se calibraron
    # los umbrales de sensibilidad. No cambiar sin recalibrar.
    mask = _harmonic_mask(f0_frames, freqs, sr, n_fft, n_harmonics, width_cents)

    # Paso 1 — Tramos de lead: "explained" mide qué parte de la energía cubre
    # la nota seguida. En rasgueos es baja (~0.15–0.3); durante un solo sube
    # aunque la rítmica siga sonando. Se promedia en ventanas largas para
    # decidir por secciones, no frame a frame.
    _report(progress, "Detectando secciones con guitarra lead…")
    band = ((freqs[:, 0] >= fmin * 0.9) & (freqs[:, 0] <= 6000.0))[:, None]
    power = (spec_raw ** 2) * band
    explained = (mask * power).sum(axis=0) / (power.sum(axis=0) + 1e-12)
    fps = sr / hop_length
    section_score = uniform_filter1d(explained.astype(np.float64), size=max(1, int(section_seconds * fps)))
    threshold = SENSITIVITY_THRESHOLDS.get(sensitivity, SENSITIVITY_THRESHOLDS["media"])
    sections = section_score >= threshold
    sections = _clean_runs(sections, min_on=int(min_section_seconds * fps), max_gap=int(max_gap_seconds * fps))
    if not sections.any():
        raise ValueError("No se encontraron secciones con guitarra lead. "
                         "Prueba con sensibilidad Alta.")

    # Paso 2 — Dentro de las secciones se extrae la nota seguida en todo frame
    # sonoro, aunque la rítmica suene fuerte; solo se omiten los frames donde
    # claramente no hay línea melódica (acorde puro).
    explained_smooth = medfilt(explained.astype(np.float64), 5)
    lead_frames = sections & (gain > 0) & (explained_smooth >= frame_threshold)
    lead_ratio = float(sections.mean())
    gain = gaussian_filter1d(lead_frames.astype(np.float32), sigma=2.0)
    _report(progress, f"Lead detectada en {lead_ratio:.0%} de la canción "
                      f"({_format_runs(sections, fps)}).")

    # Máscara de extracción: colador más ancho (bends/vibrato), extendida un
    # frame antes y después (transiciones de nota) y "decidida": lo que parece
    # lead va casi entero a la lead en vez de repartirse a medias.
    mask = _harmonic_mask(f0_extract, freqs, sr, n_fft, n_harmonics, extract_width_cents)
    mask = maximum_filter1d(mask, size=3, axis=1)
    mask = np.clip((mask - 0.15) / 0.45, 0.0, 1.0)
    mask *= gain[None, :]
    # Suavizado temporal leve para evitar "chisporroteo" en los bordes de nota.
    mask = gaussian_filter1d(mask, sigma=1.0, axis=1)
    mask = np.clip(mask, 0.0, 1.0)

    _report(progress, "Reconstruyendo guitar_lead.wav y guitar_rhythm.wav…")
    lead = _resynth(stfts, mask, hop_length, length)
    rhythm = _resynth(stfts, 1.0 - mask, hop_length, length)
    if normalize_lead:
        peak = max(float(np.max(np.abs(ch))) for ch in lead)
        if peak > 1e-6:
            lead = [ch * (0.89 / peak) for ch in lead]  # ≈ -1 dBFS
    return [
        _write(destination / "guitar_lead.wav", lead, sr),
        _write(destination / "guitar_rhythm.wav", rhythm, sr),
    ]
