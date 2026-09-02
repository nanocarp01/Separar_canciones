from __future__ import annotations

from pathlib import Path
from typing import Callable

import librosa
import numpy as np
import soundfile as sf
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF


def separate_guitars_experimental(
    input_path: str | Path,
    output_dir: str | Path,
    n_components: int = 16,
    hop_length: int = 1024,
    n_fft: int = 4096,
    random_state: int = 0,
    progress: Callable[[str], None] | None = None,
) -> list[Path]:
    """Separa un stem de guitarra en dos salidas experimentales locales."""
    source = Path(input_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if not source.exists() or source.stat().st_size == 0:
        raise ValueError("El archivo de guitarra no existe o está vacío.")
    if not 4 <= n_components <= 64:
        raise ValueError("Los componentes NMF deben estar entre 4 y 64.")

    def report(text: str) -> None:
        if progress:
            progress(text)

    report("Leyendo guitarra y calculando espectrograma…")
    audio, sr = librosa.load(str(source), sr=None, mono=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[None, :]
    elif audio.ndim != 2:
        raise ValueError(f"Formato de audio inesperado: {audio.shape}")
    # librosa con mono=False devuelve (canales, muestras).
    if audio.shape[0] > 8 and audio.shape[1] <= 8:
        audio = audio.T
    length = audio.shape[1]
    mono = np.mean(audio, axis=0)
    if not np.isfinite(mono).all() or np.max(np.abs(mono)) < 1e-8:
        raise ValueError("La pista de guitarra no contiene señal audible.")

    channels_stft = [librosa.stft(ch, n_fft=n_fft, hop_length=hop_length) for ch in audio]
    mono_stft = librosa.stft(mono, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.maximum(np.abs(mono_stft).astype(np.float32), 1e-8)
    freq_bins, frames = magnitude.shape

    report(f"Descomponiendo espectro en {n_components} componentes NMF…")
    # NMF siempre factoriza X=(muestras, características) en
    # W=(muestras, componentes) y H=(componentes, características).
    # Para audio usamos X=(frames, frecuencias), así W es temporal y H espectral.
    model = NMF(
        n_components=n_components,
        init="nndsvda",
        solver="mu",
        beta_loss="kullback-leibler",
        max_iter=300,
        random_state=random_state,
    )
    temporal = np.asarray(model.fit_transform(magnitude.T), dtype=np.float32)
    spectral = np.asarray(model.components_, dtype=np.float32)
    # temporal=(frames, components), spectral=(components, frequencies).
    if temporal.shape != (frames, n_components):
        raise ValueError(f"Dimensiones temporales NMF inesperadas: {temporal.shape}")
    if spectral.shape != (n_components, freq_bins):
        raise ValueError(f"Dimensiones espectrales NMF inesperadas: {spectral.shape}")

    report("Agrupando componentes en dos guitarras…")
    component_features = np.column_stack((
        np.log1p(np.sum(spectral, axis=1)),
        np.argmax(spectral, axis=1),
        np.mean(temporal, axis=0),
        np.std(temporal, axis=0),
        np.max(temporal, axis=0),
    )).astype(np.float32)
    scale = component_features.std(axis=0)
    normalized = (component_features - component_features.mean(axis=0)) / (scale + 1e-8)
    labels = KMeans(n_clusters=2, n_init=10, random_state=random_state).fit_predict(normalized)
    if len(np.unique(labels)) < 2:
        labels[np.argsort(np.mean(temporal, axis=0))[n_components // 2:]] = 1

    # Reconstruct energy con forma (frecuencias, frames).
    group_energy = np.zeros((2, freq_bins, frames), dtype=np.float32)
    for component, group in enumerate(labels):
        group_energy[int(group)] += np.outer(spectral[component], temporal[:, component]).astype(np.float32)
    masks = group_energy / (group_energy.sum(axis=0, keepdims=True) + 1e-8)
    masks = np.clip(masks, 0.0, 1.0)

    report("Reconstruyendo y guardando guitar_1.wav y guitar_2.wav…")
    outputs: list[Path] = []
    for group in (0, 1):
        reconstructed = []
        for stft in channels_stft:
            signal = librosa.istft(
                stft * masks[group],
                hop_length=hop_length,
                length=length,
            )
            signal = np.asarray(signal, dtype=np.float32)[:length]
            if signal.size < length:
                signal = np.pad(signal, (0, length - signal.size))
            reconstructed.append(signal)
        result = np.stack(reconstructed, axis=1)
        peak = float(np.max(np.abs(result)))
        if peak > 0.99:
            result *= 0.98 / peak
        output = destination / f"guitar_{group + 1}.wav"
        sf.write(str(output), result, sr)
        outputs.append(output)
    return outputs
