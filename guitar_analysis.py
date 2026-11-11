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
    import matplotlib.patches  # noqa: F401  (usado en _chord_figure)
    import matplotlib.ticker  # noqa: F401
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


# Colores del tema oscuro de la app (main.py)
_FIG_BG = "#1e1e2e"
_FIG_PANEL = "#27273a"
_FIG_TEXT = "#f5f5f5"
_FIG_MUTED = "#b7b7c9"
_FIG_ACCENTS = ("#7c3aed", "#2563eb")
# Ancho de la figura por segundo de audio (la ventana tiene scroll horizontal).
_INCHES_PER_SECOND = 0.6


def _chord_figure(beat_chroma, edges, segments):
    """Figura ancha: franja de acordes arriba y mapa cromático abajo."""
    duration = float(edges[-1]) if len(edges) else 1.0
    width = max(14.0, duration * _INCHES_PER_SECOND)
    fig, (ax_ch, ax) = plt.subplots(
        2, 1, figsize=(width, 4.6), sharex=True,
        gridspec_kw={"height_ratios": [1, 3], "hspace": 0.05},
    )
    fig.patch.set_facecolor(_FIG_BG)

    # Franja de acordes: bloques alternando color y altura para que no se pisen.
    ax_ch.set_facecolor(_FIG_PANEL)
    ax_ch.set_ylim(0, 2)
    ax_ch.set_yticks([])
    min_label_s = 0.45  # tramos más cortos no llevan texto
    for i, (t0, t1, ch) in enumerate(segments):
        row = i % 2
        ax_ch.add_patch(matplotlib.patches.Rectangle(
            (t0, row + 0.08), t1 - t0, 0.84, facecolor=_FIG_ACCENTS[row],
            edgecolor=_FIG_BG, linewidth=1.0))
        if t1 - t0 >= min_label_s:
            ax_ch.text((t0 + t1) / 2, row + 0.5, ch, ha="center", va="center",
                       fontsize=10, fontweight="bold", color="white", clip_on=True)
    for spine in ax_ch.spines.values():
        spine.set_visible(False)
    ax_ch.set_title("Línea de tiempo de acordes", color=_FIG_TEXT, loc="left", fontsize=12)

    ax.pcolormesh(edges, np.arange(13), beat_chroma, cmap="magma",
                  vmin=0.0, vmax=1.0, shading="flat")
    ax.set_yticks(np.arange(12) + 0.5)
    ax.set_yticklabels(NOTES)
    ax.set_facecolor(_FIG_BG)
    ax.set_xlim(0, duration)
    ax.set_xlabel("Tiempo", color=_FIG_MUTED)
    ax.set_ylabel("Nota", color=_FIG_MUTED)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda s, _: _mmss(s)))
    ax.tick_params(colors=_FIG_MUTED, labelbottom=True)
    for spine in ax.spines.values():
        spine.set_color(_FIG_PANEL)
    fig.subplots_adjust(left=0.6 / width, right=1 - 0.2 / width, top=0.9, bottom=0.14)
    return fig


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

        # Segmentación por pulsos (beats). `fix_frames` deja los límites
        # enteros, únicos, crecientes y agrega 0 y n_frames en los extremos,
        # así cada columna sincronizada j abarca [bounds[j], bounds[j+1]).
        _, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop)
        beats = np.asarray(beats)
        if beats.size < 2:  # último recurso: cortes uniformes
            beats = np.linspace(0, max(n_frames - 1, 1), num=40)
        bounds = librosa.util.fix_frames(np.round(beats).astype(int), x_min=0, x_max=n_frames)
        beat_chroma = librosa.util.sync(chroma, bounds, aggregate=np.median, pad=False)
        edges = librosa.frames_to_time(bounds, sr=sr, hop_length=hop)  # len = columnas + 1
        n = beat_chroma.shape[1]
        edges = edges[: n + 1]

        # Correlación con plantillas normalizadas
        templates = _chord_templates()
        names = list(templates.keys())
        T = np.stack([templates[k] for k in names]).astype(float)
        T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
        C = beat_chroma / (np.linalg.norm(beat_chroma, axis=0, keepdims=True) + 1e-9)
        scores = T @ C

        # Suavizado: promedio de puntajes en una ventana de 3 pulsos, para no
        # cambiar de acorde en cada pulso por ruido.
        kernel = np.ones(3) / 3.0
        smooth = np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="same"), 1, scores)
        idx = list(np.argmax(smooth, axis=0))

        # Pulsos en silencio (el croma normalizado del silencio es ruido): sin acorde.
        rms = librosa.feature.rms(y=y, hop_length=hop)
        beat_rms = librosa.util.sync(rms, bounds, aggregate=np.mean, pad=False)[0][:n]
        silent = beat_rms < (beat_rms.max() + 1e-12) * 10 ** (-40 / 20)
        for j in np.flatnonzero(silent):
            idx[j] = -1
        beat_chroma = beat_chroma * (~silent)[None, :]

        # Duración mínima: tramos de 1 pulso se absorben en el acorde vecino
        # con mejor puntaje en ese pulso.
        def runs(seq):
            out, start = [], 0
            for i in range(1, len(seq) + 1):
                if i == len(seq) or seq[i] != seq[start]:
                    out.append((start, i))
                    start = i
            return out

        min_beats = 2
        changed = True
        while changed:
            changed = False
            for start, end in runs(idx):
                if idx[start] == -1 or end - start >= min_beats or len(idx) <= min_beats:
                    continue
                left = idx[start - 1] if start > 0 else None
                right = idx[end] if end < len(idx) else None
                options = [c for c in (left, right) if c is not None and c != -1]
                if not options:
                    continue
                best = max(options, key=lambda c: smooth[c, start:end].sum())
                idx[start:end] = [best] * (end - start)
                changed = True
                break

        segments = [(float(edges[s]), float(edges[e]), names[idx[s]])
                    for s, e in runs(idx) if idx[s] != -1]
        if not segments:
            return "⚠️ No se detectó señal suficiente para reconocer acordes.", None

        text = "🎸 **Progresión de acordes detectada** — " \
               f"{len(segments)} sección(es):\n\n" + "\n".join(
                   f"[{_mmss(t0)} – {_mmss(t1)}]  **{ch}**"
                   for t0, t1, ch in segments
               )

        fig = _chord_figure(beat_chroma, edges, segments)
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
