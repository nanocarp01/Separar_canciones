"""
Análisis y transcripción de guitarra:

  • ``estimate_tuning``  — estimación de afinación (f0 con librosa.pyin).
  • ``detect_chords``    — línea de tiempo de acordes (croma + plantillas).
  • ``audio_to_midi``    — transcripción Audio → MIDI con Basic Pitch (Spotify).

Módulo independiente de la interfaz: las funciones reciben rutas de archivo y
devuelven texto / figura matplotlib / rutas de salida.
"""

from __future__ import annotations

import logging
import os
import traceback

try:
    import numpy as np
except Exception:  # noqa: BLE001
    np = None  # type: ignore[assignment]

# Backend headless para crear la figura en el worker; Tk la presenta después.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    import librosa.display
except Exception:  # noqa: BLE001
    matplotlib = None  # type: ignore[assignment]
    plt = None  # type: ignore[assignment]
    FuncFormatter = None  # type: ignore[assignment]

try:
    import librosa
except Exception:  # noqa: BLE001
    librosa = None  # type: ignore[assignment]

logger = logging.getLogger("Guitarra")

# Notas para la detección de acordes / afinación
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
E2_HZ = librosa.note_to_hz("E2") if librosa is not None else 82.41


def _mmss(sec: float) -> str:
    """Convierte segundos a formato mm:ss."""
    sec = max(0, int(round(sec)))
    return f"{sec // 60}:{sec % 60:02d}"


# ---------------------------------------------------------------------------
# Afinación
# ---------------------------------------------------------------------------
def estimate_tuning(audio_path: str) -> str:
    """Estima la afinación detectando la frecuencia fundamental más grave."""
    try:
        if librosa is None or np is None:
            return "❌ Falta instalar librosa/numpy para analizar afinación."
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        if y.size < sr * 2:
            return "⚠️ Audio demasiado corto para estimar afinación."

        f0, voiced, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("E4"),
            sr=sr,
        )
        freqs = f0[voiced]
        if len(freqs) < 10:
            return "⚠️ No se detectaron notas graves suficientes para estimar la afinación."

        # Frecuencia grave sostenida (percentil 15 de las f0 detectadas)
        low_freq = float(np.percentile(freqs, 15))
        cents = 1200.0 * np.log2(low_freq / E2_HZ)

        tunings = [
            ("Afinación Estándar (E A D G B E)", 0),
            ("Medio tono abajo (E♭ A♭ D♭ G♭ B♭ E♭)", -100),
            ("Un tono abajo / Drop D (D A D G B E)", -200),
            ("Medio tono arriba (F A# D# G# C F)", 100),
            ("Un tono arriba (F# B E A C# F#)", 200),
        ]
        best, offset = min(tunings, key=lambda t: abs(cents - t[1]))

        if abs(cents - offset) > 30:
            verdict = f"🎯 Afinación **no estándar** (desviación de {cents:+.0f} cents respecto a E2)"
        else:
            verdict = f"🎯 Afinación probable: **{best}**"

        return (
            verdict
            + f"\n📉 Cuerda grave detectada: **{low_freq:.1f} Hz** "
            + f"(referencia E2 = {E2_HZ:.1f} Hz · desviación {cents:+.0f} cents)"
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Error estimando afinación:\n%s", traceback.format_exc())
        return f"❌ Error estimando la afinación: {exc}"


# ---------------------------------------------------------------------------
# Acordes
# ---------------------------------------------------------------------------
def _chord_templates() -> dict[str, np.ndarray]:
    """Plantillas de acordes (mayor, menor, 7ª dominante y disminuido)."""
    base = {
        "":    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        "m":   [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        "7":   [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        "dim": [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    }
    templates: dict[str, np.ndarray] = {}
    for root in range(12):
        for suffix, vec in base.items():
            templates[f"{NOTES[root]}{suffix}"] = np.roll(vec, root)
    return templates


def detect_chords(audio_path: str) -> tuple[str, object]:
    """Detecta una línea de tiempo de acordes.

    Devuelve ``(texto_markdown, figura_matplotlib_o_None)``.
    """
    try:
        if librosa is None or np is None or plt is None:
            return "❌ Faltan librosa, numpy o matplotlib para detectar acordes.", None
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        if y.size < sr * 2:
            return "⚠️ Audio demasiado corto para detectar acordes.", None

        hop = 512
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
        n_frames = chroma.shape[1]

        # Segmentación por pulsos (beats). Se acotan/limpian SIEMPRE a índices
        # de frame válidos (enteros, únicos, crecientes y < n_frames) para que
        # librosa.util.sync y el bucle de abajo nunca fallen por desborde.
        _, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop)
        beats = np.asarray(beats)
        if beats.size < 2:
            beats = np.arange(0, n_frames, max(1, n_frames // 40))
        beats = np.unique(np.round(beats).astype(int))
        beats = beats[(beats >= 0) & (beats < n_frames)]
        if beats.size < 2:  # último recurso: cortes uniformes
            beats = np.unique(np.linspace(0, max(n_frames - 1, 1), num=40).astype(int))

        # `librosa.util.sync` con pad=True agrega UNA columna extra al final
        # (el tramo entre el último beat y el final del audio), así que el
        # resultado tiene len(beats)+1 columnas. Alineamos `times` agregando
        # el frame final para que ambas tengan la misma longitud; de lo
        # contrario, con ciertos audios el bucle de abajo intenta acceder a
        # times[len(labels)] y lanza IndexError (out of bounds).
        beat_chroma = librosa.util.sync(chroma, beats, aggregate=np.median)
        times = librosa.frames_to_time(
            np.append(beats, n_frames), sr=sr, hop_length=hop
        )

        # Correlación con plantillas normalizadas
        templates = _chord_templates()
        names = list(templates.keys())
        T = np.stack([templates[n] for n in names]).astype(float)
        T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
        C = beat_chroma / (np.linalg.norm(beat_chroma, axis=0, keepdims=True) + 1e-9)
        scores = T @ C
        labels = [names[i] for i in np.argmax(scores, axis=0)]

        # Fusiona acordes consecutivos iguales en secciones (índices seguros)
        n = len(labels)
        segments, start = [], 0
        for i in range(1, n + 1):
            if i == n or labels[i] != labels[start]:
                t0 = times[min(start, n - 1)]
                t1 = times[min(i - 1, n - 1)]
                if i - 1 <= start:
                    t1 = t0 + 1.0
                segments.append((t0, max(t0 + 1.0, t1), labels[start]))
                start = i

        text = "🎸 **Progresión de acordes detectada** — " \
               f"{len(segments)} sección(es):\n\n" + "\n".join(
                   f"[{_mmss(t0)} – {_mmss(t1)}]  **{ch}**"
                   for t0, t1, ch in segments
               )

        # Figura: mapa cromático + etiquetas de acorde
        fig, ax = plt.subplots(figsize=(14, 3.6))
        librosa.display.specshow(
            beat_chroma, x_coords=times, y_axis="chroma",
            ax=ax, cmap="magma", vmin=0.0, vmax=1.0,
        )
        for t0, t1, ch in segments:
            ax.text(
                (t0 + t1) / 2, 11.6, ch, ha="center", va="bottom",
                fontsize=11, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.25", fc="#7c3aed", ec="none", alpha=0.9),
            )
        ax.set_title("Mapa cromático por pulso (línea de tiempo de acordes)")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda s, _: _mmss(s)))
        fig.tight_layout()
        return text, fig

    except Exception as exc:  # noqa: BLE001
        logger.error("Error detectando acordes:\n%s", traceback.format_exc())
        return f"❌ Error detectando acordes: {exc}", None


# ---------------------------------------------------------------------------
# Audio → MIDI
# ---------------------------------------------------------------------------
def _load_basic_pitch_model():
    """Carga el modelo de Basic Pitch priorizando GPU (CUDA) y cayendo a CPU.

    Usa onnxruntime para la inferencia. Intenta primero cargar el modelo con
    CUDAExecutionProvider (requiere ``onnxruntime-gpu`` y CUDA instalados); si
    no está disponible, falla la creación de la sesión o no hay provider CUDA,
    se usa CPUExecutionProvider.

    Devuelve ``(model, backend)`` donde backend es 'cuda' o 'cpu'.
    """
    import basic_pitch
    from basic_pitch.inference import Model

    model_path = basic_pitch.ICASSP_2022_MODEL_PATH

    # --- Intento primero con GPU ---
    try:
        import onnxruntime as ort

        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("onnxruntime-gpu no instalado")

        # Pre-carga las DLLs de CUDA/cuDNN. Si torch (CUDA) está instalado, sus
        # DLLs (cuda12/cudnn9) ya vienen en su carpeta lib; preload_dlls las
        # busca y las expone a onnxruntime. Esto es OBLIGATORIO para que el
        # provider CUDA enlace con el cuDNN/CUDA correcto.
        try:
            import torch._C  # noqa: F401  # fuerza cargar DLLs CUDA de torch
        except Exception:  # noqa: BLE001
            pass
        try:
            # onnxruntime-gpu>=1.21 expone preload_dlls para precargar
            # CUDA, cuDNN y MSVC desde la instalación de PyTorch o NVIDIA.
            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls()
        except Exception as exc:  # noqa: BLE001
            logger.warning("preload_dlls falló (%s); se intenta igualmente.", exc)

        session = ort.InferenceSession(
            str(model_path),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        # Verifica que CUDA se esté usando de verdad (no solo que esté listado)
        used = session.get_providers()
        if "CUDAExecutionProvider" not in used:
            raise RuntimeError(f"CUDA no activo en la sesión (usando {used})")

        # Model.__init__ con la ruta construye y asigna self.model_type en ONNX,
        # aunque fuerza CPU en su sesión interna; le sustituimos esa sesión por
        # la nuestra con CUDA ya activado.
        model = Model(model_path)
        model.model = session  # reemplaza la sesión interna CPU por la CUDA
        logger.info("Basic Pitch: usando GPU (CUDAExecutionProvider)")
        return model, "cuda"
    except Exception as exc:  # noqa: BLE001
        logger.warning("Basic Pitch con GPU no disponible (%s); usando CPU.", exc)

    # --- Fallback a CPU ---
    try:
        return Model(model_path), "cpu"
    except Exception as exc:  # noqa: BLE001
        logger.error("No se pudo cargar Basic Pitch ni en CPU: %s", exc)
        raise


def audio_to_midi(audio_path: str) -> tuple[object, str]:
    """Transcribe el audio a MIDI con Basic Pitch (Spotify),
    priorizando GPU (CUDA) y cayendo a CPU si no hay.

    Devuelve ``(ruta_del_midi_o_None, mensaje)``.
    """
    try:
        from basic_pitch.inference import predict

        model, backend = _load_basic_pitch_model()
        logger.info("Transcribiendo a MIDI (Basic Pitch) en %s...", backend)
        _, midi_data, note_events = predict(audio_path, model)
        out_path = os.path.join(os.path.dirname(audio_path), "transcripcion_guitarra.mid")
        midi_data.write(out_path)
        n = len(note_events)
        logger.info("MIDI generado con %d notas: %s", n, out_path)
        dev = "🚀 GPU CUDA" if backend == "cuda" else "🐢 CPU"
        return out_path, f"✅ MIDI generado con **{n} nota(s)** ({dev}) — compatible con TuxGuitar / Guitar Pro."
    except Exception as exc:  # noqa: BLE001
        logger.error("Error en transcripción MIDI:\n%s", traceback.format_exc())
        return None, f"❌ Error en la transcripción MIDI: {exc}"
