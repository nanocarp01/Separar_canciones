#!/usr/bin/env python3
"""
Demucs · RoFormer + Análisis de Guitarra — Audio Source Separator (Tkinter)

Aplicación de escritorio organizada en **3 pestañas**:

  Tab 1 · 🎛️ Separador       — separa la canción en stems (Demucs 4/6 o
                                RoFormer/MDXC) con GPU CUDA y fallback CPU.
  Tab 2 · 🎚️ Multitrack     — carga automáticamente lo separado y permite
                                reproducir, **silenciar (mute)**, **solo** y
                                ajustar el **volumen** de cada pista, y
                                generar una mezcla WAV.
  Tab 3 · 🎸 Análisis        — estimación de **afinación**, **acordes** y
                                transcripción **Audio → MIDI** (Basic Pitch).

Toda la separación corre en un hilo secundario; la GUI nunca se congela.
"""

from __future__ import annotations

import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import traceback
from pathlib import Path
from typing import Optional

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from ffmpeg_utils import ensure_ffmpeg_in_path
from guitar_analysis import (
    estimate_tuning,
    detect_chords,
    audio_to_midi,
)
from midside import MIDSIDE_PRESETS, apply_midside_filter

# Mapea la etiqueta del Combobox (españa UI) a la clave interna del filtro
MIDSIDE_LABEL_TO_KEY = {label: key for key, label, _hint in MIDSIDE_PRESETS}

try:
    import torch  # noqa: F401
    TORCH_AVAILABLE = True
    TORCH_IMPORT_ERROR = ""
except Exception as exc:  # noqa: BLE001
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False
    TORCH_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

try:
    import torchaudio  # noqa: F401
    TORCHAUDIO_AVAILABLE = True
    TORCHAUDIO_IMPORT_ERROR = ""
except Exception as exc:  # noqa: BLE001
    torchaudio = None  # type: ignore[assignment]
    TORCHAUDIO_AVAILABLE = False
    TORCHAUDIO_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

try:
    import soundfile as sf  # noqa: F401
    SOUNDFILE_AVAILABLE = True
    SOUNDFILE_IMPORT_ERROR = ""
except Exception as exc:  # noqa: BLE001
    sf = None  # type: ignore[assignment]
    SOUNDFILE_AVAILABLE = False
    SOUNDFILE_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

# Motor de reproducción en vivo (pestaña multitrack)
try:
    from playback import PlaybackEngine, SOUNDDEVICE_AVAILABLE  # noqa: F401
except Exception as exc:  # noqa: BLE001
    PlaybackEngine = None  # type: ignore[assignment]
    SOUNDDEVICE_AVAILABLE = False

# Demucs (motor 4 / 6 fuentes)
DemucsSeparator = None  # type: ignore[assignment]
DEMUCS_AVAILABLE = False
DEMUCS_IMPORT_ERROR = ""
try:
    from demucs.api import Separator as _DemucsSeparator  # type: ignore
    DemucsSeparator = _DemucsSeparator
    DEMUCS_AVAILABLE = True
except Exception as exc:  # noqa: BLE001
    DEMUCS_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

# audio-separator (motor RoFormer / MDXC / Demucs vía UVR)
AudioSeparator = None  # type: ignore[assignment]
ASEP_AVAILABLE = False
ASEP_IMPORT_ERROR = ""
try:
    from audio_separator.separator import Separator as _AudioSeparator  # type: ignore
    AudioSeparator = _AudioSeparator
    ASEP_AVAILABLE = True
except Exception as exc:  # noqa: BLE001
    ASEP_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
APP_TITLE = "Separador Musical + Guitarra"
APP_VERSION = "3.0"
PROGRAM_FOLDER = "Separador_Musical"


def _music_dir() -> Path:
    """Carpeta Música real de Windows (respeta redirección a OneDrive).

    Usa SHGetKnownFolderPath con FOLDERID_Music; si falla, cae a ~/Music.
    """
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class _GUID(ctypes.Structure):
                _fields_ = [
                    ("Data1", wintypes.DWORD),
                    ("Data2", wintypes.WORD),
                    ("Data3", wintypes.WORD),
                    ("Data4", ctypes.c_ubyte * 8),
                ]

            # FOLDERID_Music: {4BD8D571-6D19-48D3-BE97-422220080E43}
            folder_id = _GUID(
                0x4BD8D571, 0x6D19, 0x48D3,
                (ctypes.c_ubyte * 8)(0xBE, 0x97, 0x42, 0x22, 0x20, 0x08, 0x0E, 0x43),
            )
            fn = ctypes.windll.shell32.SHGetKnownFolderPath
            fn.argtypes = [
                ctypes.POINTER(_GUID), wintypes.DWORD, wintypes.HANDLE,
                ctypes.POINTER(ctypes.c_wchar_p),
            ]
            fn.restype = wintypes.HRESULT
            buf = ctypes.c_wchar_p()
            if fn(ctypes.byref(folder_id), 0, None, ctypes.byref(buf)) == 0 and buf.value:
                return Path(buf.value)
    except Exception:  # noqa: BLE001
        pass
    return Path.home() / "Music"


DEFAULT_OUTPUT_DIR = _music_dir() / PROGRAM_FOLDER


def _sanitize_folder_name(name: str) -> str:
    """Quita caracteres inválidos para carpetas de Windows."""
    name = re.sub(r"[<>:\"/\\|?*\x00-\x1f]", "_", name).strip().rstrip(". ")
    return name[:120] or "tema"
SUPPORTED_EXTS = (".mp3", ".wav", ".flac", ".ogg", ".m4a")

STEM_LABELS = {
    "vocals":  "🎤 Voz",
    "drums":   "🥁 Batería",
    "bass":    "🎸 Bajo",
    "other":   "🎧 Otros",
    "guitar":  "🎸 Guitarra",
    "piano":   "🎹 Piano",
}

# Modelos RoFormer / MDXC curados que pueden elegirse desde la GUI.
ASEP_CURATED_MODELS: list[tuple[str, str, str]] = [
    (
        "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        "BS-RoFormer · vocals/instrumental",
        "Por defecto. Separa voz vs instrumental con calidad alta.",
    ),
    (
        "htdemucs_6s.yaml",
        "Demucs v4 · 6 fuentes (vía UVR)",
        "Mismo htdemucs_6s, expuesto por audio-separator.",
    ),
    (
        "htdemucs_ft.yaml",
        "Demucs v4 · 4 fuentes (fine-tuned)",
        "Versión fine-tuned de 4 fuentes, mejor calidad SDR.",
    ),
    (
        "MDX23C-DrumSep-aufr33-jarredou.ckpt",
        "MDX23C DrumSep · 6 partes de batería",
        "kick / snare / toms / hh / ride / crash — útil para batería detallada.",
    ),
]


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


def column_stack_2(arr) -> "np.ndarray":  # type: ignore[name-defined]
    import numpy as np
    return np.column_stack((arr, arr))


# ===========================================================================
# Aplicación Tkinter
# ===========================================================================
class AudioSeparatorApp:
    MODE_DEMUCS_4 = "4 Fuentes (Estándar) — Demucs htdemucs"
    MODE_DEMUCS_6 = "6 Fuentes (Guitarra/Piano) — Demucs htdemucs_6s"
    MODE_ROFORMER = "RoFormer / MDXC — audio-separator (GPU)"

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(f"{APP_TITLE}  v{APP_VERSION}")
        self.root.geometry("860x640")
        self.root.minsize(780, 560)

        # Estado interno
        self.input_file: Optional[str] = None
        self.output_dir: Path = DEFAULT_OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.message_queue: "queue.Queue[tuple[str, object]]" = queue.Queue()
        self.worker_thread: Optional[threading.Thread] = None
        self.is_processing: bool = False

        # Stems separados (para la pestaña multitrack)
        self.stems: dict[str, str] = {}       # nombre interno -> ruta wav
        self.last_run_out: Optional[str] = None

        # Controles del mixer (rellenos por _build_mixer_row)
        self.mixer_rows: dict[str, dict] = {}

        # Reproducción en vivo (pestaña multitrack)
        self.playback: Optional[PlaybackEngine] = None
        if PlaybackEngine is not None:
            try:
                self.playback = PlaybackEngine()
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                self.playback = None
        self._active_track: Optional[str] = None
        self.master_var = tk.DoubleVar(value=100.0)

        # Análisis de guitarra
        self.analysis_audio: Optional[str] = None

        # Bandera para avisar una sola vez si corre en CPU teniendo GPU
        self._warned_cpu = True

        self._build_ui()
        self._check_environment()
        self.root.after(120, self._poll_queue)

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        # Fuentes compartidas guardadas como atributos (los métodos de tab las usan)
        self._f_title = (
            ("Segoe UI", 16, "bold")
            if sys.platform == "win32"
            else ("Helvetica", 16, "bold")
        )
        self._f_sub = ("Segoe UI", 10) if sys.platform == "win32" else ("Helvetica", 10)
        self._f_btn = ("Segoe UI", 12, "bold") if sys.platform == "win32" else ("Helvetica", 12, "bold")
        self._f_body = ("Segoe UI", 10) if sys.platform == "win32" else ("Helvetica", 10)

        # Cabecera
        header = ttk.Frame(self.root)
        header.pack(fill="x", padx=16, pady=(12, 4))
        ttk.Label(header, text="🎵  Separador Musical + Guitarra",
                  font=self._f_title).pack(anchor="w")
        ttk.Label(
            header,
            text=("Demucs · RoFormer · librosa · Basic Pitch — usa la RTX 5070 "
                  "cuando esté disponible."),
            font=self._f_sub, foreground="#5f6368",
        ).pack(anchor="w")

        # Notebook con las 3 pestañas
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=12, pady=(4, 0))

        self._build_separar_tab()
        self._build_multitrack_tab()
        self._build_guitarra_tab()

        # Footer
        self.env_label = ttk.Label(self.root, text="", font=self._f_body)
        self.env_label.pack(side="bottom", fill="x", padx=12, pady=4)

        self._on_mode_changed()

    # ---------------------------------------------------- Tab 1 · Separar
    def _build_separar_tab(self) -> None:
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="🎛️  1 · Separar Pistas")

        # 1) Archivo
        file_frame = ttk.LabelFrame(tab, text=" Archivo de audio ", padding=10)
        file_frame.pack(fill="x", padx=12, pady=(12, 6))
        self.file_label = ttk.Label(
            file_frame, text="Ningún archivo seleccionado",
            foreground="gray", wraplength=520, font=("Segoe UI", 10),
        )
        self.file_label.pack(side="left", fill="x", expand=True)
        ttk.Button(file_frame, text="📁  Cargar",
                   command=self._on_load_file).pack(side="right")

        # 2) Motor
        mode_frame = ttk.LabelFrame(tab, text=" Motor de separación ", padding=10)
        mode_frame.pack(fill="x", padx=12, pady=4)
        ttk.Label(mode_frame, text="Motor:").pack(side="left", padx=(0, 8))
        self.mode_var = tk.StringVar(value=self.MODE_DEMUCS_4)
        self.mode_combo = ttk.Combobox(
            mode_frame, textvariable=self.mode_var,
            values=self._available_modes(), state="readonly", width=46,
        )
        self.mode_combo.pack(side="left")
        self.mode_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_mode_changed())
        self.mode_desc = ttk.Label(mode_frame, text="", foreground="#5f6368")
        self.mode_desc.pack(side="left", padx=(12, 0))

        self.sub_frame = ttk.Frame(tab)
        self.sub_label = ttk.Label(self.sub_frame, text="Modelo:")
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(
            self.sub_frame, textvariable=self.model_var, state="readonly", width=60,
        )
        self.model_var2_desc = ttk.Label(self.sub_frame, text="", foreground="#5f6368")
        self.model_combo.bind(
            "<<ComboboxSelected>>", lambda _e: self._update_model_desc()
        )

        # 3) Post-proc
        post_frame = ttk.LabelFrame(tab, text=" Post-procesamiento extra ", padding=10)
        post_frame.pack(fill="x", padx=12, pady=4)
        self.midside_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            post_frame,
            text=("Separación estéreo Mid/Side/L/R sobre guitar.wav o piano.wav"),
            variable=self.midside_var,
        ).pack(anchor="w")
        ttk.Label(
            post_frame,
            text="Útil cuando el tema tiene guitarras paneadas a lados distintos.",
            foreground="#5f6368",
        ).pack(anchor="w", pady=(2, 0))

        # Filtro Mid/Side para limpiar el público de la voz (temas en vivo)
        self.clean_vocals_var = tk.BooleanVar(value=False)
        clean_row = ttk.Frame(post_frame)
        clean_row.pack(anchor="w", pady=(8, 0))
        ttk.Checkbutton(
            clean_row,
            text="Limpiar público de la voz (filtro Mid/Side / karaoke)",
            variable=self.clean_vocals_var,
        ).pack(side="left")
        self.clean_vocals_mode_var = tk.StringVar(value=MIDSIDE_PRESETS[0][1])
        ttk.Combobox(
            clean_row,
            textvariable=self.clean_vocals_mode_var,
            state="readonly",
            width=38,
            values=[label for _key, label, _hint in MIDSIDE_PRESETS],
        ).pack(side="left", padx=(8, 0))
        ttk.Label(
            post_frame,
            text="Para directos: el cantante está al centro y el público en los "
                 "laterales. 'Conservar centro' quita el público estéreo de la voz.",
            foreground="#5f6368",
        ).pack(anchor="w", pady=(2, 0))

        # Estado + botón
        status_frame = ttk.LabelFrame(tab, text=" Estado ", padding=10)
        status_frame.pack(fill="x", padx=12, pady=4)
        self.status_label = ttk.Label(status_frame, text="Listo para separar.",
                                      font=("Segoe UI", 10))
        self.status_label.pack(anchor="w")
        self.progress = ttk.Progressbar(status_frame, mode="indeterminate")
        self.progress.pack(fill="x", pady=(8, 0))

        self.start_btn = ttk.Button(
            tab, text="▶   Empezar Separación", command=self._on_start,
        )
        self.start_btn.pack(pady=(10, 8), ipadx=24, ipady=8)
        try:
            self.start_btn.configure(font=self._f_btn)
        except tk.TclError:
            pass

        # Aviso de que pasa al multitrack
        ttk.Label(
            tab,
            text=("Al terminar, las pistas pasan a la pestaña 2 · Multitrack "
                  "para reproducirlas y mezclarlas."),
            foreground="#5f6368", font=self._f_sub,
        ).pack(pady=(0, 8))

    # --------------------------------------------------- Tab 2 · Multitrack
    def _build_multitrack_tab(self) -> None:
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="🎚️  2 · Multitrack")

        self.mt_header = ttk.Label(
            tab, text="Aún no hay pistas separadas. Ve a la pestaña 1 y separa una canción.",
            foreground="gray", wraplength=560,
        )
        self.mt_header.pack(anchor="w", padx=12, pady=(12, 6))

        # Contenedor de filas de stems (se llena al separar)
        self.mt_canvas_frame = ttk.Frame(tab, padding=6)
        self.mt_canvas_frame.pack(fill="both", expand=True, padx=12)

        # Datos para la mezcla
        self.mt_info = ttk.Label(tab, text="", foreground="#5f6368", wraplength=580)
        self.mt_info.pack(anchor="w", padx=12, pady=(4, 0))

        # Reproductor maestro (en vivo dentro de la app)
        master_frame = ttk.LabelFrame(tab, text=" Reproductor maestro ", padding=8)
        master_frame.pack(fill="x", padx=12, pady=8)
        ttk.Button(master_frame, text="▶ Play", width=6,
                   command=self._on_master_play).pack(side="left", padx=2)
        ttk.Button(master_frame, text="⏸ Pausa", width=6,
                   command=self._on_master_pause).pack(side="left", padx=2)
        ttk.Button(master_frame, text="⏹ Detener", width=7,
                   command=self._on_master_stop).pack(side="left", padx=2)
        ttk.Label(master_frame, text="  Master:").pack(side="left", padx=(12, 4))
        ttk.Scale(master_frame, from_=0, to=150, variable=self.master_var,
                  command=self._on_master_volume, length=160).pack(side="left")
        self.master_val_lbl = ttk.Label(master_frame, text="100%", width=6)
        self.master_val_lbl.pack(side="left")
        self.now_playing_lbl = ttk.Label(master_frame, text="Nada sonando.",
                                         foreground="#5f6368", wraplength=220)
        self.now_playing_lbl.pack(side="left", padx=(12, 0))

        # Controles de mezcla
        mix_frame = ttk.LabelFrame(tab, text=" Mezcla ", padding=10)
        mix_frame.pack(fill="x", padx=12, pady=8)
        self.mix_btn = ttk.Button(
            mix_frame, text="🎧  Generar y Reproducir Mezcla",
            command=self._on_make_mix, state="disabled",
        )
        self.mix_btn.pack(side="left")
        self.mix_status = ttk.Label(mix_frame, text="", foreground="#2e7d32")
        self.mix_status.pack(side="left", padx=(12, 0))
        ttk.Button(mix_frame, text="⬇ Descargar mezcla (.wav)",
                   command=self._on_save_mix).pack(side="right")

    # ------------------------------------------------ Tab 3 · Análisis
    def _build_guitarra_tab(self) -> None:
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="🎸  3 · Análisis y Transcripción")

        row = ttk.Frame(tab)
        row.pack(fill="x", padx=12, pady=(12, 4))
        self.guitar_label = ttk.Label(
            row,
            text="Elige un WAV de guitarra (p. ej. guitar.wav de la separación) o "
                 "usa el que se cargó de la Tab 2.",
            foreground="gray", wraplength=460,
        )
        self.guitar_label.pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="🎸 Cargar Guitarra",
                   command=self._on_load_guitar).pack(side="right")

        # Afinación
        tune_frame = ttk.LabelFrame(tab, text=" 1 · Afinación estimada ", padding=8)
        tune_frame.pack(fill="x", padx=12, pady=4)
        self.tuning_out = ttk.Label(tune_frame, text="—", wraplength=620, justify="left")
        self.tuning_out.pack(anchor="w")

        # Acordes
        chord_frame = ttk.LabelFrame(tab, text=" 2 · Acordes de la guitarra ", padding=8)
        chord_frame.pack(fill="x", padx=12, pady=4)
        self.chords_out = tk.Text(
            chord_frame, height=8, wrap="word", state="disabled",
            bg="#f6f6f6", font=("Segoe UI", 10),
        )
        self.chords_out.pack(fill="x")
        ttk.Label(
            chord_frame,
            text="La línea de tiempo de acordes también se muestra como imagen en una "
                 "ventana aparte (botón Ver Gráfico).",
            foreground="#5f6368",
        ).pack(anchor="w", pady=(4, 0))
        self.show_chords_btn = ttk.Button(
            chord_frame, text="📈 Ver Gráfico de Acordes",
            command=self._on_show_chords, state="disabled",
        )
        self.show_chords_btn.pack(anchor="w", pady=(4, 0))

        # Transcripción MIDI
        midi_frame = ttk.LabelFrame(tab, text=" 3 · Transcripción Audio → MIDI ", padding=8)
        midi_frame.pack(fill="x", padx=12, pady=4)
        self.midi_btn = ttk.Button(midi_frame, text="🎹 Generar MIDI",
                                   command=self._on_midi, state="disabled")
        self.midi_btn.pack(side="left")
        self.midi_status_label = ttk.Label(midi_frame, text="", foreground="#2e7d32",
                                           wraplength=480)
        self.midi_status_label.pack(side="left", padx=(12, 0))

        # Botón análisis general
        self.analyze_btn = ttk.Button(
            tab, text="🔍 Analizar Guitarra (Afinación + Acordes)",
            command=self._on_analyze_guitar, state="disabled",
        )
        self.analyze_btn.pack(pady=(8, 8), ipadx=12, ipady=6)

        self._last_chords_fig = None

    # ------------------------------------------------- Modos de separación

    def _available_modes(self) -> list[str]:
        modes = []
        if DEMUCS_AVAILABLE:
            modes.append(self.MODE_DEMUCS_4)
            modes.append(self.MODE_DEMUCS_6)
        if ASEP_AVAILABLE:
            modes.append(self.MODE_ROFORMER)
        return modes or [self.MODE_DEMUCS_4]

    def _on_mode_changed(self) -> None:
        mode = self.mode_var.get()
        if mode == self.MODE_DEMUCS_4:
            self.mode_desc.config(text="(htdemucs — vocals/drums/bass/other)")
            self._hide_sub_model()
        elif mode == self.MODE_DEMUCS_6:
            self.mode_desc.config(text="(htdemucs_6s — vocals/drums/bass/other/guitar/piano)")
            self._hide_sub_model()
        elif mode == self.MODE_ROFORMER:
            self.mode_desc.config(text="(BS-RoFormer, MDXC o Demucs vía audio-separator)")
            self._show_sub_model()
        else:
            self.mode_desc.config(text="")
            self._hide_sub_model()

    def _show_sub_model(self) -> None:
        self.sub_frame.pack(fill="x", padx=12, pady=(0, 2))
        self.sub_label.pack(side="left", padx=(0, 8))
        if not self.model_combo["values"]:
            self._refresh_model_list()
        self.model_combo.pack(side="left")
        self.model_var2_desc.pack(side="left", padx=(12, 0))
        self._update_model_desc()

    def _hide_sub_model(self) -> None:
        self.sub_frame.pack_forget()

    def _refresh_model_list(self) -> None:
        if not ASEP_AVAILABLE:
            return
        try:
            sep = AudioSeparator()  # type: ignore[misc]
            catalog = sep.get_simplified_model_list() or {}
        except Exception:  # noqa: BLE001
            catalog = {}
        curated_filenames = {fn for fn, _label, _desc in ASEP_CURATED_MODELS}
        ordered: list[tuple[str, str, str]] = list(ASEP_CURATED_MODELS)
        for fn, info in catalog.items():
            if fn in curated_filenames:
                continue
            name = info.get("Name") or fn
            stems = info.get("Stems") or []
            label = f"{name}  ({fn})"
            desc = ", ".join(s for s in stems if isinstance(s, str))
            ordered.append((fn, label, desc))
        self._asep_cached_models = ordered
        self.model_combo["values"] = [label for _fn, label, _desc in ordered]
        if ordered:
            self.model_var.set(ordered[0][1])

    def _selected_model(self) -> tuple[str, str]:
        cached = getattr(self, "_asep_cached_models", ASEP_CURATED_MODELS)
        label = self.model_var.get()
        for fn, lbl, desc in cached:
            if lbl == label:
                return fn, desc
        return ASEP_CURATED_MODELS[0][0], ASEP_CURATED_MODELS[0][2]

    def _update_model_desc(self) -> None:
        if not ASEP_AVAILABLE:
            self.model_var2_desc.config(text="audio-separator no disponible")
            return
        _fn, desc = self._selected_model()
        if desc:
            self.model_var2_desc.config(text=f"({desc[:80]}{'…' if len(desc) > 80 else ''})")

    # --------------------------------------------------------- Estado GUI

    def _check_environment(self) -> None:
        ffmpeg_dir = ensure_ffmpeg_in_path()

        if not TORCH_AVAILABLE:
            self.env_label.config(
                text=f"⚠ PyTorch no instalado.  {TORCH_IMPORT_ERROR}",
                foreground="#c62828",
            )
            self.start_btn.config(state="disabled")
            return
        if not (DEMUCS_AVAILABLE or ASEP_AVAILABLE):
            self.env_label.config(
                text=("⚠ No hay motor de separación disponible.\n"
                      f"  Demucs: {DEMUCS_IMPORT_ERROR}\n"
                      f"  audio-separator: {ASEP_IMPORT_ERROR}"),
                foreground="#c62828",
            )
            self.start_btn.config(state="disabled")
            return

        if not DEMUCS_AVAILABLE:
            self.env_label.config(
                text=f"⚠ Demucs no disponible: {DEMUCS_IMPORT_ERROR}",
                foreground="#c62828",
            )
        if not ASEP_AVAILABLE:
            self.env_label.config(
                text=f"⚠ audio-separator no disponible: {ASEP_IMPORT_ERROR}",
                foreground="#c62828",
            )

        gpu_msg = ""
        if cuda_available():
            gpu_msg = f"✓ GPU: {gpu_name() or 'CUDA'} · "
        else:
            gpu_msg = "ℹ CPU only · "

        ffmpeg_msg = (
            f"ffmpeg OK ({Path(ffmpeg_dir).name}) · " if ffmpeg_dir
            else "⚠ ffmpeg NO encontrado (instalación necesaria) · "
        )

        if cuda_available():
            self.env_label.config(
                text=gpu_msg + ffmpeg_msg + "Prioridad GPU en separación y MIDI.",
                foreground="#2e7d32",
            )
        else:
            # Muestra build de torch para diagnosticar (p. ej. +cpu vs +cu128)
            torch_ver = str(getattr(torch, "__version__", "?")) if torch is not None else "?"
            torch_build = torch_ver.split("+")[-1] if "+" in torch_ver else torch_ver
            # Intérprete real que ejecuta el script (para orientar el diagnóstico)
            py_path = os.path.dirname(sys.executable)
            msg = (gpu_msg + ffmpeg_msg
                   + f"Sin GPU (torch {torch_ver}). "
                   + "Ejecutas run.bat o el python de: "
                   + f"{py_path}\n\n"
                   + "Si tienes NVIDIA, cierra y usa run.bat "
                   + "(lanza el .venv de este proyecto, el que tiene CUDA).")
            self.env_label.config(text=msg, foreground="#ef6c00")
            if self._warned_cpu:  # avisar una sola vez por sesión
                messagebox.showwarning(
                    "Corriendo con CPU",
                    "Se está ejecutando con torch SIN CUDA (build '{}').\n\n"
                    "Tienes una GPU NVIDIA pero no se está usando.\n\n"
                    "El intérprete actual es:\n    {}\n\n"
                    "Esto ocurre cuando se ejecuta con el Python del PATH o de "
                    "otro venv (sin el torch con CUDA de este proyecto).\n\n"
                    "Cierra esta ventana y abre la app con:\n"
                    "    run.bat\n"
                    "(usa .venv con torch +cuXX de Separar_canciones)".format(
                        torch_build, py_path,
                    )
                )
                self._warned_cpu = False

        if not SOUNDDEVICE_AVAILABLE:
            # Avisar una vez (best effort) si no se puede reproducir en la app
            self._set_mix_info(
                "ℹ No se detectó 'sounddevice' para reproducir en la app; "
                "las pistas se abrirán en el reproductor externo."
            )

        if ASEP_AVAILABLE and not ffmpeg_dir:
            messagebox.showwarning(
                "FFmpeg no encontrado",
                "No se encontró ffmpeg.exe en el sistema.\n\n"
                "El motor RoFormer / MDXC lo necesita para decodificar MP3, FLAC, etc.\n\n"
                "Instálalo con:  winget install -e --id Gyan.FFmpeg\n"
                "O agrega la carpeta que contiene ffmpeg.exe al PATH.",
            )

    # ------------------------------------------------------------- Eventos

    def _on_load_file(self) -> None:
        if self.is_processing:
            messagebox.showinfo(
                "Procesando", "Hay una separación en curso. Espera a que termine."
            )
            return
        path = filedialog.askopenfilename(
            title="Seleccionar archivo de audio",
            filetypes=[
                ("Archivos de audio", " ".join(f"*{e}" for e in SUPPORTED_EXTS)),
                ("MP3", "*.mp3"),
                ("WAV", "*.wav"),
                ("FLAC", "*.flac"),
                ("Todos los archivos", "*.*"),
            ],
        )
        if not path:
            return
        ext = Path(path).suffix.lower()
        if ext not in SUPPORTED_EXTS:
            messagebox.showwarning(
                "Formato no soportado",
                f"La extensión '{ext}' no está en la lista recomendada; "
                "se intentará igualmente.",
            )
        self.input_file = path
        self.file_label.config(text=path, foreground="black")

    def _on_start(self) -> None:
        if self.is_processing:
            return
        if not self.input_file:
            messagebox.showwarning("Sin archivo", "Carga primero un archivo de audio.")
            return

        mode = self.mode_var.get()
        if mode == self.MODE_ROFORMER and not ASEP_AVAILABLE:
            messagebox.showerror(
                "audio-separator no disponible", f"Motivo: {ASEP_IMPORT_ERROR}"
            )
            return
        if mode != self.MODE_ROFORMER and not DEMUCS_AVAILABLE:
            messagebox.showerror(
                "Demucs no disponible", f"Motivo: {DEMUCS_IMPORT_ERROR}"
            )
            return

        if mode == self.MODE_ROFORMER and not cuda_available():
            if not messagebox.askyesno(
                "RoFormer en CPU",
                "No se detectó GPU con CUDA y elegiste RoFormer / MDXC.\n\n"
                "En CPU es muy lento. ¿Continuar de todas formas?",
            ):
                return

        self.is_processing = True
        self.start_btn.config(state="disabled")
        self.progress.start(15)
        self.status_label.config(text="Iniciando…", foreground="#1565c0")

        kwargs = {
            "input_path": self.input_file,
            "midside": self.midside_var.get(),
            "clean_vocals": self.clean_vocals_var.get(),
            "clean_vocals_mode": MIDSIDE_LABEL_TO_KEY.get(
                self.clean_vocals_mode_var.get(), "center"
            ),
        }
        if mode == self.MODE_DEMUCS_4:
            kwargs["engine"] = "demucs"
            kwargs["engine_model"] = "htdemucs"
        elif mode == self.MODE_DEMUCS_6:
            kwargs["engine"] = "demucs"
            kwargs["engine_model"] = "htdemucs_6s"
        elif mode == self.MODE_ROFORMER:
            kwargs["engine"] = "asep"
            kwargs["engine_model"], _ = self._selected_model()
        else:
            messagebox.showerror("Error", "Modo no soportado")
            self.is_processing = False
            self.start_btn.config(state="normal")
            self.progress.stop()
            return

        self.worker_thread = threading.Thread(
            target=self._run_separation, kwargs=kwargs, daemon=True
        )
        self.worker_thread.start()

    # -------------------------------------------------- Hilo de trabajo

    def _run_separation(
        self, input_path: str, engine: str, engine_model: str, midside: bool,
        clean_vocals: bool = False, clean_vocals_mode: str = "center",
    ) -> None:
        try:
            # Validación temprana: un archivo de 0 bytes (descarga/copia
            # incompleta) hace que Demucs/RoFormer fallen con errores crípticos
            # de sphn/ffmpeg. Mejor avisar claro antes de arrancar.
            src = Path(input_path)
            if not src.exists():
                raise RuntimeError(
                    f"El archivo no existe:\n{input_path}"
                )
            if src.stat().st_size == 0:
                raise RuntimeError(
                    "El archivo está VACÍO (0 bytes) y no se puede leer.\n\n"
                    f"📄 {src.name}\n\n"
                    "Probablemente la descarga o la copia quedó incompleta.\n"
                    "Volvé a descargar el archivo y probá de nuevo."
                )

            track_stem = _sanitize_folder_name(Path(input_path).stem)
            run_out = self.output_dir / track_stem
            run_out.mkdir(parents=True, exist_ok=True)

            if engine == "demucs":
                self._run_demucs(input_path, engine_model, run_out)
            elif engine == "asep":
                self._run_asep(input_path, engine_model, run_out)
            else:
                raise RuntimeError(f"Motor desconocido: {engine}")

            if clean_vocals:
                self.message_queue.put((
                    "status",
                    "[Mid/Side] Limpiando el público de la pista de voz…",
                ))
                self._clean_vocals_stem(run_out, clean_vocals_mode)

            if midside:
                self.message_queue.put(
                    ("status", "Post-procesando pistas estéreo (Mid/Side/L/R)…")
                )
                self._apply_midside(run_out)

            saved = sorted(p for p in run_out.iterdir() if p.suffix.lower() == ".wav")
            self.message_queue.put(("done", (str(run_out), saved)))

        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            msg = str(exc)
            # Traduce los errores de carga de audio de Demucs (sphn/ffmpeg) a
            # un mensaje comprensible en lugar del traceback críptico.
            low = msg.lower()
            if ("loadaudioerror" in low or "no suitable format reader" in low
                    or "could not read the file" in low
                    or "ffmpeg could not" in low):
                msg = (
                    "No se pudo leer el archivo de audio: está vacío, dañado "
                    "o en un formato no soportado.\n\n"
                    "Verificá que el archivo no tenga 0 bytes (descarga "
                    "incompleta) y que sea MP3/WAV/FLAC/OGG válido.\n\n"
                    f"📄 {Path(input_path).name}\n\nDetalle técnico: {msg}"
                )
            self.message_queue.put(("error", f"{msg}\n\nDetalles:\n{tb}"))

    def _run_demucs(self, input_path: str, model_name: str, run_out: Path) -> None:
        device = "cuda" if cuda_available() else "cpu"
        self.message_queue.put((
            "status",
            f"[Demucs] Cargando '{model_name}' en {device}… "
            "(primera vez puede tardar en descargar pesos)",
        ))
        separator = DemucsSeparator(model=model_name, device=device)
        self.message_queue.put(("status", f"[Demucs] '{model_name}' separando…"))
        _origin, separated = separator.separate_audio_file(input_path)
        self.message_queue.put(("status", "[Demucs] Guardando archivos WAV…"))
        sample_rate = getattr(separator, "samplerate", 44100)
        self._save_dict_as_wav(separated, run_out, sample_rate)

    def _run_asep(self, input_path: str, model_filename: str, run_out: Path) -> None:
        import tempfile

        with tempfile.TemporaryDirectory(prefix="asep_") as tmp:
            tmp_path = Path(tmp)
            sep = AudioSeparator(  # type: ignore[misc]
                output_dir=str(tmp_path), output_format="WAV",
            )
            dev = sep.setup_accelerated_inferencing_device()
            self.message_queue.put((
                "status",
                f"[RoFormer] Acelerador: {dev}  ·  Cargando '{model_filename}'…",
            ))
            sep.load_model(model_filename=model_filename)
            self.message_queue.put(("status", "[RoFormer] Ejecutando separación…"))
            out_files = sep.separate(input_path)  # tipo: list[str]

            moved = 0
            for f in out_files:
                p = Path(f)
                src = p if p.is_absolute() else (tmp_path / p.name)
                if not src.exists():
                    alt = tmp_path / p.name
                    if alt.exists():
                        src = alt
                    else:
                        self.message_queue.put((
                            "status", f"[RoFormer] WARN: source no encontrado: {src}",
                        ))
                        continue
                new_name = self._clean_asep_name(src.name)
                if new_name is None:
                    continue
                dst = run_out / new_name
                if dst.exists():
                    dst.unlink()
                shutil.move(str(src), str(dst))
                moved += 1

            self.message_queue.put((
                "status",
                f"[RoFormer] Guardados {moved} stem(s) en {run_out.name}",
            ))

    @staticmethod
    def _clean_asep_name(filename: str) -> Optional[str]:
        m = re.search(r"\(([^)]+)\)", filename)
        if not m:
            return None
        stem = m.group(1).strip().lower().replace(" ", "_")
        return f"{stem}.wav"

    def _save_dict_as_wav(self, separated: dict, folder: Path, sr: int) -> list[Path]:
        if sf is None:
            raise RuntimeError("soundfile no está disponible para escribir WAV.")
        folder.mkdir(parents=True, exist_ok=True)
        saved = []
        for name, wave in separated.items():
            w = wave.detach().cpu()
            if w.ndim == 3:
                w = w[0]
            data = w.numpy().T.astype("float32", copy=False)
            out_path = folder / f"{name}.wav"
            sf.write(str(out_path), data, sr)
            saved.append(out_path)
        return saved

    # -------------------------------------------------- Post Mid/Side

    def _clean_vocals_stem(self, folder: Path, mode: str) -> None:
        """Aplica el filtro Mid/Side a la pista de voz, reemplazando el WAV."""
        target: Optional[Path] = None
        for name in ("vocals.wav", "vocal.wav"):
            cand = folder / name
            if cand.exists():
                target = cand
                break
        if target is None:
            self.message_queue.put((
                "status",
                "[Mid/Side] No se encontró la pista de voz para limpiar.",
            ))
            return
        try:
            apply_midside_filter(str(target), str(target), mode=mode)
            self.message_queue.put((
                "status",
                f"[Mid/Side] Voz limpia aplicada ({mode}).",
            ))
        except Exception as exc:  # noqa: BLE001
            self.message_queue.put((
                "status",
                f"[Mid/Side] No se pudo limpiar la voz: {exc}",
            ))

    def _apply_midside(self, folder: Path) -> list[Path]:
        if sf is None:
            return []
        produced: list[Path] = []
        for target in ("guitar.wav", "piano.wav"):
            src = folder / target
            if not src.exists():
                continue
            data, sr = sf.read(str(src), always_2d=True)
            if data.shape[1] < 2:
                continue
            left = data[:, 0]
            right = data[:, 1]
            mid = 0.5 * (left + right)
            side = 0.5 * (left - right)
            stem = target[:-4]
            for label, channel in (("L", left), ("R", right),
                                   ("Mid", mid), ("Side", side)):
                out = folder / f"{stem}_{label}.wav"
                if label in ("L", "R", "Mid", "Side"):
                    arr = column_stack_2(channel)
                else:
                    arr = channel
                sf.write(str(out), arr, sr)
                produced.append(out)
        return produced

    # =================================================== MULTITRACK (Tab 2)
    # ------------------------------------------------- Carga automática
    def _populate_mixer(self, run_out: str, files) -> None:
        """Rellena la pestaña Multitrack con una fila por stem."""
        # Limpia filas anteriores
        for _name, row in list(self.mixer_rows.items()):
            try:
                row["frame"].destroy()
            except Exception:  # noqa: BLE001
                pass
        self.mixer_rows.clear()

        # Detecta stems por nombre normalizado. Resuelve rutas absolutas o
        # relativas (basename) contra run_out para ser robusto.
        run_out = Path(run_out)
        stems: dict[str, str] = {}
        for f in files:
            p = Path(f)
            # Si no es absoluto y no existe tal cual, asume que vive en run_out
            if not p.is_absolute() or not p.exists():
                cand = run_out / p.name
                if cand.exists():
                    p = cand
            stem_name = self._normalize_stem(p.stem)
            if stem_name:
                stems[stem_name] = str(p)

        if not stems:
            self.mt_header.config(
                text="No se reconocieron pistas en esta carpeta.",
                foreground="#c62828",
            )
            return

        self.stems = stems
        self.last_run_out = run_out

        self.mt_header.config(
            text=f"Pistas separadas — {len(stems)} · ▶ reproduce en la app, "
                 f"silencia (M), solo (S) y ajusta volumen de cada una.",
            foreground="#2e7d32",
        )

        # Crea una fila por stem (orden estable)
        order = ["vocals", "drums", "bass", "other", "guitar", "piano"]
        for stem_name in order:
            if stem_name not in stems:
                continue
            self._build_mixer_row(stem_name, stems[stem_name])

        self.mix_btn.config(state="normal")
        self._set_mix_info("Usa mute/solo/volumen y pulsa 'Generar y Reproducir Mezcla'.")

    def _normalize_stem(self, base: str) -> Optional[str]:
        """Convierte 'guitar', 'Guitar', 'vocals' etc. a nombre interno."""
        b = base.lower()
        for key in ("guitar", "piano", "vocals", "vocals_", "vocal", "drums",
                    "bass", "other", "instrumental", "accompaniment"):
            if b.startswith(key):
                # 'instrumental'/'accompaniment' -> other
                if key in ("instrumental", "accompaniment"):
                    return "other"
                return key
        return None

    def _build_mixer_row(self, name: str, path: str) -> None:
        frame = ttk.Frame(self.mt_canvas_frame, padding=4)
        frame.pack(fill="x", pady=2)

        label = STEM_LABELS.get(name, name.capitalize())
        ttk.Label(frame, text=label, width=14).pack(side="left")

        # Volumen
        vol = tk.DoubleVar(value=100.0)
        scale = ttk.Scale(frame, from_=0, to=150, variable=vol, length=180,
                          command=lambda v: self._on_volume(name, v))
        scale.pack(side="left", padx=(6, 4))
        val_lbl = ttk.Label(frame, text="100%", width=6)
        val_lbl.pack(side="left")

        # Mute / Solo
        mute = tk.BooleanVar(value=False)
        solo = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="M", width=3, variable=mute,
                        command=lambda: self._on_mute_changed()).pack(side="left", padx=2)
        ttk.Checkbutton(frame, text="S (Solo)", variable=solo,
                        command=lambda: self._on_solo_changed()).pack(side="left", padx=2)

        # Reproducción en vivo dentro de la app
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(side="left", padx=6)
        ttk.Button(btn_frame, text="▶", width=3,
                   command=lambda n=name: self._play_track(n)).pack(side="left", padx=1)
        ttk.Button(btn_frame, text="⏸", width=3,
                   command=lambda n=name: self._pause_track(n)).pack(side="left", padx=1)
        ttk.Button(btn_frame, text="⏹", width=3,
                   command=lambda n=name: self._stop_track(n)).pack(side="left", padx=1)
        ttk.Button(btn_frame, text="🎛️", width=3,
                   command=lambda n=name: self._on_midside_filter(n),
                   ).pack(side="left", padx=1)

        self.mixer_rows[name] = {
            "frame": frame, "vol": vol, "val_lbl": val_lbl, "path": path,
            "mute": mute, "solo": solo, "checked": [mute, solo],
        }

    # ------------------------- Filtro Mid/Side por pista (karaoke / público)
    def _on_midside_filter(self, name: str) -> None:
        """Diálogo para aplicar cancelación de fase (Mid/Side) a una pista."""
        row = self.mixer_rows.get(name)
        if not row:
            return
        label = STEM_LABELS.get(name, name.capitalize())

        win = tk.Toplevel(self.root)
        win.title(f"Filtro Mid/Side — {label}")
        win.transient(self.root)
        win.grab_set()
        win.resizable(False, False)

        ttk.Label(
            win,
            text=(
                "Cancelación de fase (Vocal Pan / Mid-Side).\n\n"
                "En temas en vivo el cantante está al CENTRO (aparece igual en "
                "L y R) y el público suele estar en los LATERALES. Elegí el "
                "modo y guardá el que mejor suene:"
            ),
            wraplength=470, justify="left",
        ).pack(anchor="w", padx=14, pady=(12, 6))

        var = tk.StringVar(value="center")
        for key, mode_label, hint in MIDSIDE_PRESETS:
            ttk.Radiobutton(win, text=mode_label, value=key, variable=var,
                            ).pack(anchor="w", padx=14)
            ttk.Label(win, text=hint, foreground="#5f6368", wraplength=460,
                      justify="left").pack(anchor="w", padx=(28, 14))

        def _apply() -> None:
            mode = var.get()
            # Verifica que sea estéreo antes de ofrecer sobrescribir
            try:
                info = sf.read(row["path"], always_2d=True, dtype="float32")
                if info.shape[1] < 2:
                    messagebox.showerror(
                        "Pista mono",
                        "Esta pista es MONO y la cancelación de fase necesita "
                        "estéreo (canales L/R). No se puede aplicar.",
                    )
                    return
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror(
                    "Filtro Mid/Side", f"No se pudo leer la pista:\n{exc}"
                )
                return

            if not messagebox.askyesno(
                "Sobrescribir",
                f"Se va a reemplazar el WAV de '{label}' con la versión "
                f"filtrada ({mode}).\n\n¿Continuar?",
            ):
                return
            win.destroy()
            try:
                self._stop_track(name)
                apply_midside_filter(row["path"], row["path"], mode=mode)
                self._set_mix_info(
                    f"🎛️ Filtro Mid/Side '{mode}' aplicado a {label} y guardado. "
                    "Regenerá la mezcla si la necesitás."
                )
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                messagebox.showerror(
                    "Filtro Mid/Side", f"No se pudo aplicar el filtro:\n{exc}"
                )

        btns = ttk.Frame(win)
        btns.pack(pady=10)
        ttk.Button(btns, text="✓ Aplicar y guardar", command=_apply
                   ).pack(side="left", padx=4)
        ttk.Button(btns, text="Cancelar", command=win.destroy).pack(side="left", padx=4)

    def _on_volume(self, name: str, _val: str) -> None:
        row = self.mixer_rows.get(name)
        if row:
            pct = int(round(row["vol"].get()))
            row["val_lbl"].config(text=f"{pct}%")
            # Aplica el volumen en vivo si esta pista está sonando
            if (self.playback is not None and self._active_track == name
                    and (self.playback.is_playing() or self.playback.is_paused())):
                self.playback.set_volume(pct / 100.0)

    def _on_mute_changed(self) -> None:
        self._set_mix_info("Mezcla previsualizada: recuerda pulsar 'Generar y Reproducir'.")

    def _on_solo_changed(self) -> None:
        self._set_mix_info("Mezcla previsualizada: recuerda pulsar 'Generar y Reproducir'.")

    # --------------------------- Reproducción en vivo (in-app)
    def _play_track(self, name: str) -> None:
        row = self.mixer_rows.get(name)
        if not row:
            return
        path = row["path"]
        if self.playback is None or not SOUNDDEVICE_AVAILABLE:
            # Fallback: reproductor externo si no hay sounddevice
            self._play_file_external(path)
            return
        try:
            self._active_track = name
            self.playback.play(
                path,
                volume=row["vol"].get() / 100.0,
                master=self.master_var.get() / 100.0,
            )
            self._set_mix_info(f"▶ Sonando en vivo: {STEM_LABELS.get(name, name)} · "
                               f"usa ⏸ pausa, ⏹ detener o el master aquí abajo.")
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            messagebox.showerror("Reproducir", f"No se pudo reproducir en vivo:\n{exc}")

    def _pause_track(self, name: str) -> None:
        if self.playback is None:
            return
        if self._active_track is not None and (
            (self._active_track == name) or name is None
        ):
            self.playback.pause()
            self._set_mix_info("⏸ Pausado.")

    def _stop_track(self, name: str) -> None:
        if self.playback is None:
            return
        if self._active_track is not None and (
            self._active_track == name or name is None
        ):
            self.playback.stop()
            self._active_track = None
            self._set_mix_info("⏹ Detenido.")

    # ------------------------------ Controles maestro
    def _on_master_volume(self, _val: str) -> None:
        pct = int(round(self.master_var.get()))
        self.master_val_lbl.config(text=f"{pct}%")
        if self.playback is not None:
            self.playback.set_master(pct / 100.0)

    def _on_master_play(self) -> None:
        if self.playback is None:
            messagebox.showinfo("Reproductor", "No hay motor de audio disponible.")
            return
        if self.playback.is_paused():
            self.playback.resume()
            return
        # Reproduce desde la primera pista si no hay nada activo
        if self._active_track is None and self.mixer_rows:
            # Respeta mute/solo: elige la primera pista audible
            target = None
            any_solo = any(r["solo"].get() for r in self.mixer_rows.values())
            for name, row in self.mixer_rows.items():
                if row["mute"].get():
                    continue
                if any_solo and not row["solo"].get():
                    continue
                target = name
                break
            if target:
                self._play_track(target)

    def _on_master_pause(self) -> None:
        if self.playback is not None:
            self.playback.pause()

    def _on_master_stop(self) -> None:
        if self.playback is not None:
            self.playback.stop()
            self._active_track = None

    def _play_file_external(self, path: str) -> None:
        """Reproduce con el reproductor del sistema (fallback sin sounddevice)."""
        try:
            if sys.platform == "win32":
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Reproducir", f"No se pudo reproducir:\n{exc}")

    def _set_mix_info(self, text: str) -> None:
        self.mt_info.config(text=text, foreground="#1565c0")

    # ------------------------------------------------- Mezcla
    def _active_gains(self) -> dict[str, float]:
        """Calcula la ganancia efectiva (0-1.5) considerando mute y solo."""
        any_solo = any(r["solo"].get() for r in self.mixer_rows.values())
        gains: dict[str, float] = {}
        for name, row in self.mixer_rows.items():
            if row["mute"].get():
                gains[name] = 0.0
            elif any_solo and not row["solo"].get():
                gains[name] = 0.0
            else:
                gains[name] = row["vol"].get() / 100.0
        return gains

    def _mix_path(self) -> Optional[str]:
        if not self.stems or not self.last_run_out:
            return None
        return os.path.join(self.last_run_out, "mezcla_multitrack.wav")

    def _on_make_mix(self) -> None:
        if not self.stems:
            messagebox.showinfo("Sin pistas", "Primero separa una canción.")
            return
        try:
            gains = self._active_gains()
            out_path = self._mix_path()
            self._render_mix(out_path, gains)
            self.mix_status.config(
                text=f"✅ Mezcla creada. ({'GPU/CPU' if cuda_available() else 'CPU'})",
                foreground="#2e7d32",
            )
            # Reproduce la mezcla en vivo dentro de la app
            if self.playback is not None and SOUNDDEVICE_AVAILABLE:
                self._active_track = None  # la mezcla no es una pista individual
                self.playback.play(
                    out_path, volume=1.0, master=self.master_var.get() / 100.0,
                )
                self._set_mix_info("🎧 Reproduciendo la mezcla en vivo...")
            else:
                self._play_file_external(out_path)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            messagebox.showerror("Mezcla", f"Error al mezclar:\n{exc}")

    def _render_mix(self, out_path: str, gains: dict[str, float]) -> None:
        if sf is None:
            raise RuntimeError("soundfile no disponible.")
        mixed, sr = None, None
        for name, path in self.stems.items():
            if not os.path.exists(path):
                continue
            y, s = sf.read(path, always_2d=True)
            g = gains.get(name, 1.0)
            mixed = y * g if mixed is None else mixed + y * g
            sr = s
        if mixed is None:
            raise RuntimeError("No hay pistas para mezclar.")
        import numpy as _np
        mixed = _np.clip(mixed, -1.0, 1.0)  # evita saturación
        sf.write(out_path, mixed, sr)

    def _on_save_mix(self) -> None:
        src = self._mix_path()
        if not src or not os.path.exists(src):
            messagebox.showinfo("Sin mezcla", "Primero genera la mezcla.")
            return
        dst = filedialog.asksaveasfilename(
            title="Guardar mezcla", defaultextension=".wav",
            initialfile="mezcla_multitrack.wav",
            filetypes=[("WAV", "*.wav"), ("Todos los archivos", "*.*")],
        )
        if not dst:
            return
        try:
            shutil.copyfile(src, dst)
            messagebox.showinfo("Mezcla guardada", f"Guardado en:\n{dst}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Guardar", str(exc))

    # ================================================= GUITARRA (Tab 3)
    # ------------------------------------------------- Carga automática
    def _load_guitar_from_mixer(self) -> None:
        """Carga automáticamente guitar.wav separado en la Tab 3 (si existe)."""
        if "guitar" in self.stems and self.stems["guitar"]:
            self.analysis_audio = self.stems["guitar"]
            self.guitar_label.config(
                text=f"Guitarra cargada automáticamente: {os.path.basename(self.stems['guitar'])}",
                foreground="#2e7d32",
            )
            self.analyze_btn.config(state="normal")
            self.midi_btn.config(state="normal")
        elif "bass" in self.stems:
            # fallback: si no hay guitar.wav, ofrecer el bass para análisis
            self.analysis_audio = self.stems["bass"]
            self.guitar_label.config(
                text=f"No hay guitar.wav; se cargó bass.wav: {os.path.basename(self.stems['bass'])}",
                foreground="#5f6368",
            )
            self.analyze_btn.config(state="normal")
            self.midi_btn.config(state="normal")

    def _on_load_guitar(self) -> None:
        if self.is_processing:
            messagebox.showinfo("Procesando", "Hay una separación en curso.")
            return
        path = filedialog.askopenfilename(
            title="Seleccionar WAV de guitarra",
            filetypes=[("WAV", "*.wav"), ("Todos los archivos", "*.*")],
        )
        if not path:
            return
        self.analysis_audio = path
        self.guitar_label.config(text=path, foreground="black")
        self.analyze_btn.config(state="normal")
        self.midi_btn.config(state="normal")
        self._set_chords("— WAV cargado. Pulsa 'Analizar Guitarra' para ver "
                         "afinación y acordes.")

    def _set_chords(self, text: str) -> None:
        self.chords_out.config(state="normal")
        self.chords_out.delete("1.0", tk.END)
        self.chords_out.insert("1.0", text)
        self.chords_out.config(state="disabled")

    def _on_analyze_guitar(self) -> None:
        if self.is_processing:
            messagebox.showinfo("Procesando", "Espera a que termine la separación.")
            return
        if not self.analysis_audio:
            messagebox.showwarning("Sin archivo", "Carga primero un WAV de guitarra.")
            return
        self.analyze_btn.config(state="disabled")
        self.root.update_idletasks()
        try:
            tuning = estimate_tuning(self.analysis_audio)
            chords, fig = detect_chords(self.analysis_audio)
            self._last_chords_fig = fig
            self.tuning_out.config(text=tuning, foreground="#2e7d32")
            self._set_chords(chords)
            # habilita el botón de gráfico
            self.show_chords_btn.config(state="normal")
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            messagebox.showerror("Error en análisis", str(exc))
        finally:
            self.analyze_btn.config(state="normal")

    def _on_show_chords(self) -> None:
        if self._last_chords_fig is not None:
            self._show_figure(self._last_chords_fig)
        else:
            messagebox.showinfo("Sin gráfico", "Primero analiza la guitarra.")

    def _on_midi(self) -> None:
        if not self.analysis_audio:
            messagebox.showwarning("Sin archivo", "Carga primero un WAV de guitarra.")
            return
        self.midi_btn.config(state="disabled")
        self.root.update_idletasks()
        try:
            path, msg = audio_to_midi(self.analysis_audio)
            self.midi_status_label.config(text=msg, foreground="#2e7d32")
            if path:
                messagebox.showinfo(
                    "MIDI generado", f"{msg}\n\nArchivo: {path}\n\n"
                    "Compatible con TuxGuitar / Guitar Pro."
                )
        finally:
            self.midi_btn.config(state="normal")

    def _show_figure(self, fig) -> None:
        import matplotlib
        matplotlib.use("TkAgg")
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        win = tk.Toplevel(self.root)
        win.title("🎸 Línea de tiempo de acordes")
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        ttk.Button(win, text="Cerrar", command=win.destroy).pack(pady=6)

    # ------------------------------------------------------- Cola UI

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self.message_queue.get_nowait()
                if kind == "status":
                    self.status_label.config(text=str(payload), foreground="#1565c0")
                elif kind == "done":
                    folder, files = payload  # type: ignore[misc]
                    self._on_finished(str(folder), files)  # type: ignore[arg-type]
                elif kind == "error":
                    self._on_error(str(payload))
        except queue.Empty:
            pass
        finally:
            self.root.after(120, self._poll_queue)

    def _on_finished(self, folder: str, files) -> None:
        self.is_processing = False
        self.start_btn.config(state="normal")
        self.progress.stop()
        names = ", ".join(Path(p).name for p in files[:6])
        more = "" if len(files) <= 6 else f"  (+{len(files) - 6} más)"
        self.status_label.config(
            text=f"✓ Separación completada — {len(files)} archivos: {names}{more}",
            foreground="#2e7d32", wraplength=720,
        )
        # Carga automática al multitrack y guitarra
        self._populate_mixer(folder, files)
        self._load_guitar_from_mixer()
        # Cambia a la pestaña multitrack para que vea las pistas
        self.notebook.select(1)
        self._open_folder(folder)

    def _on_error(self, msg: str) -> None:
        self.is_processing = False
        self.start_btn.config(state="normal")
        self.progress.stop()
        self.status_label.config(
            text="✗ Error durante la separación (ver diálogo).",
            foreground="#c62828",
        )
        messagebox.showerror("Error en la separación", msg)

    def _open_folder(self, folder: str) -> None:
        path = os.path.normpath(folder)
        try:
            if sys.platform == "win32":
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception:
            messagebox.showinfo(
                "Carpeta de salida", f"No pude abrir la carpeta automáticamente.\n\n{path}"
            )


# ===========================================================================
# Entry point
# ===========================================================================
def main() -> None:
    root = tk.Tk()
    try:
        style = ttk.Style()
        for theme in ("vista", "clam", "default"):
            if theme in style.theme_names():
                style.theme_use(theme)
                break
    except Exception:  # pragma: no cover
        pass

    AudioSeparatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()