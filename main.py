#!/usr/bin/env python3
"""
Demucs · RoFormer — Audio Source Separator (Tkinter)

Aplicación de escritorio con tres motores de separación:

  1. Demucs 4 fuentes  →  htdemucs (vocals, drums, bass, other)
  2. Demucs 6 fuentes  →  htdemucs_6s (+ guitar, piano)
  3. RoFormer / MDXC   →  audio-separator sobre BS-RoFormer / Demucs / MDXC,
                           usando GPU CUDA si está disponible

Adicionalmente puede aplicar una separación estéreo extra (Mid/Side, L, R)
sobre la guitarra para casos en que haya guitarras paneadas a lados
distintos.

Toda la separación corre en un hilo secundario; la GUI nunca se congela.
Al terminar se abre la carpeta con los WAV resultantes.
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

# ---------------------------------------------------------------------------
# FFmpeg detection
# ---------------------------------------------------------------------------
# audio-separator y demucs invocan ``subprocess.run(["ffmpeg", "-version"])``
# en cada separación. Si ffmpeg no está en el PATH (lo habitual cuando se
# instala vía winget y aún no se ha reabierto la terminal), reventaría.
# Esta función busca el binario en ubicaciones comunes y agrega el directorio
# resultante a os.environ["PATH"] del proceso.


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


def _find_ffmpeg_dir() -> Optional[str]:
    """Devuelve el directorio donde está ffmpeg.exe o None si no lo encuentra."""
    # Primero intenta el PATH actual
    found = shutil.which("ffmpeg")
    if found:
        return str(Path(found).parent)

    binary_name = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"

    # Busca nivel directo en candidatos
    for d in _FFMPEG_CANDIDATE_DIRS:
        if not d:
            continue
        candidate = Path(d) / binary_name
        if candidate.exists():
            return str(candidate.parent)

    # Búsqueda recursiva superficial (bin/ subcarpeta o ffmpeg-*-build)
    # Limitada para no ser lenta.
    for d in _FFMPEG_CANDIDATE_DIRS:
        if not d:
            continue
        base = Path(d)
        if not base.exists():
            continue
        try:
            # Mira hasta 2 niveles: dir/ o dir/subdir/bin/
            for entry in base.iterdir():  # 1 nivel
                if entry.is_file() and entry.name.lower() == binary_name:
                    return str(entry.parent)
                if entry.is_dir() and entry.name.lower() in ("bin",):
                    inner = entry / binary_name
                    if inner.exists():
                        return str(inner.parent)
                if entry.is_dir() and ("ffmpeg" in entry.name.lower()):
                    # e.g. ffmpeg-9.0.1-essentials_build/bin/ffmpeg.exe
                    inner = entry / "bin" / binary_name
                    if inner.exists():
                        return str(inner.parent)
                    inner2 = entry / binary_name
                    if inner2.exists():
                        return str(inner2.parent)
        except (PermissionError, OSError):
            continue
    return None


def _ensure_ffmpeg_in_path() -> Optional[str]:
    """Asegura que ffmpeg esté en el PATH de este proceso. Devuelve la ruta."""
    ffmpeg_dir = _find_ffmpeg_dir()
    if ffmpeg_dir is None:
        return None
    cur = os.environ.get("PATH", "")
    if ffmpeg_dir.lower() not in cur.lower().split(os.pathsep):
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + cur
    return ffmpeg_dir


# Llamar al cargar el módulo para que subprocess dentro de audio-separator
# lo encuentre tanto en CPU como en GPU.
_ensure_ffmpeg_in_path()

# ---------------------------------------------------------------------------
# Dependencias opcionales
# ---------------------------------------------------------------------------
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
APP_TITLE = "Demucs · RoFormer — Separador de Fuentes"
APP_VERSION = "2.0"
DEFAULT_OUTPUT_DIR = Path.home() / "DemucsSeparated"
SUPPORTED_EXTS = (".mp3", ".wav", ".flac", ".ogg", ".m4a")

# Modelos RoFormer / MDXC curados que pueden elegirse desde la GUI.
# (filename, etiqueta amigable, descripción corta para el usuario)
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
        self.root.geometry("820x600")
        self.root.minsize(720, 540)

        # Estado interno
        self.input_file: Optional[str] = None
        self.output_dir: Path = DEFAULT_OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.message_queue: "queue.Queue[tuple[str, object]]" = queue.Queue()
        self.worker_thread: Optional[threading.Thread] = None
        self.is_processing: bool = False

        self._build_ui()
        self._check_environment()
        self.root.after(120, self._poll_queue)

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        f_title = (
            ("Segoe UI", 16, "bold")
            if sys.platform == "win32"
            else ("Helvetica", 16, "bold")
        )
        f_sub = ("Segoe UI", 10) if sys.platform == "win32" else ("Helvetica", 10)
        f_btn = ("Segoe UI", 12, "bold") if sys.platform == "win32" else ("Helvetica", 12, "bold")
        f_body = ("Segoe UI", 10) if sys.platform == "win32" else ("Helvetica", 10)

        # Cabecera
        ttk.Label(self.root, text="🎵  Separador de Fuentes de Audio",
                  font=f_title).pack(pady=(20, 2))
        ttk.Label(
            self.root,
            text=(
                "Demucs + RoFormer — usa la RTX 5070 cuando esté disponible."
            ),
            font=f_sub,
            foreground="#5f6368",
        ).pack()

        # 1) Archivo
        file_frame = ttk.LabelFrame(self.root, text=" 1 · Archivo de audio ", padding=12)
        file_frame.pack(fill="x", padx=24, pady=(20, 6))

        self.file_label = ttk.Label(
            file_frame, text="Ningún archivo seleccionado",
            foreground="gray", wraplength=560, font=f_body,
        )
        self.file_label.pack(side="left", fill="x", expand=True)

        ttk.Button(file_frame, text="📁  Cargar Archivo",
                   command=self._on_load_file).pack(side="right")

        # 2) Motor de separación
        mode_frame = ttk.LabelFrame(self.root, text=" 2 · Motor de separación ", padding=12)
        mode_frame.pack(fill="x", padx=24, pady=6)

        ttk.Label(mode_frame, text="Motor:").pack(side="left", padx=(0, 8))
        self.mode_var = tk.StringVar(value=self.MODE_DEMUCS_4)
        self.mode_combo = ttk.Combobox(
            mode_frame,
            textvariable=self.mode_var,
            values=self._available_modes(),
            state="readonly",
            width=46,
        )
        self.mode_combo.pack(side="left")
        self.mode_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_mode_changed())
        self.mode_desc = ttk.Label(mode_frame, text="", foreground="#5f6368")
        self.mode_desc.pack(side="left", padx=(12, 0))

        # Sub-modelo RoFormer (oculto por defecto, se muestra al seleccionar RoFormer)
        self.sub_frame = ttk.Frame(self.root)
        self.sub_label = ttk.Label(self.sub_frame, text="Modelo:")
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(
            self.sub_frame,
            textvariable=self.model_var,
            state="readonly",
            width=60,
        )
        self.model_var2_desc = ttk.Label(self.sub_frame, text="", foreground="#5f6368")
        self.model_combo.bind(
            "<<ComboboxSelected>>", lambda _e: self._update_model_desc()
        )

        # 3) Post-procesamiento estéreo
        post_frame = ttk.LabelFrame(
            self.root, text=" 3 · Post-procesamiento extra ", padding=12
        )
        post_frame.pack(fill="x", padx=24, pady=6)

        self.midside_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            post_frame,
            text=(
                "Aplicar separación estéreo (Mid / Side / L / R) "
                "sobre guitar.wav o piano.wav"
            ),
            variable=self.midside_var,
        ).pack(anchor="w")

        ttk.Label(
            post_frame,
            text=(
                "Útil cuando el tema tiene guitarras paneadas a lados "
                "distintos. No distingue eléctrica de acústica."
            ),
            foreground="#5f6368",
        ).pack(anchor="w", pady=(2, 0))

        # 4) Estado
        status_frame = ttk.LabelFrame(self.root, text=" 4 · Estado ", padding=12)
        status_frame.pack(fill="x", padx=24, pady=6)

        self.status_label = ttk.Label(status_frame, text="Listo para separar.", font=f_body)
        self.status_label.pack(anchor="w")

        self.progress = ttk.Progressbar(status_frame, mode="indeterminate")
        self.progress.pack(fill="x", pady=(8, 0))

        # Botón principal
        self.start_btn = ttk.Button(
            self.root, text="▶   Empezar Separación", command=self._on_start,
        )
        self.start_btn.pack(pady=(14, 8), ipadx=24, ipady=10)
        try:
            self.start_btn.configure(font=f_btn)
        except tk.TclError:
            pass

        # Footer
        self.env_label = ttk.Label(self.root, text="", font=f_body)
        self.env_label.pack(side="bottom", pady=4)

        self._on_mode_changed()

    def _available_modes(self) -> list[str]:
        modes = []
        if DEMUCS_AVAILABLE:
            modes.append(self.MODE_DEMUCS_4)
            modes.append(self.MODE_DEMUCS_6)
        if ASEP_AVAILABLE:
            modes.append(self.MODE_ROFORMER)
        # Si ninguno está disponible, devolver una lista por defecto
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
        self.sub_frame.pack(fill="x", padx=24, pady=(0, 2))
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
        # Mezclar catalog + curados; los curados van primero
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
        # Seleccionar el primero (BS-RoFormer default)
        if ordered:
            self.model_var.set(ordered[0][1])

    def _selected_model(self) -> tuple[str, str]:
        """Devuelve (filename, desc) del modelo RoFormer elegido."""
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
        # Buscar ffmpeg ANTES de evaluar nada más (puede ser crítico)
        ffmpeg_dir = _ensure_ffmpeg_in_path()

        if not TORCH_AVAILABLE:
            self.env_label.config(
                text=f"⚠ PyTorch no instalado.  {TORCH_IMPORT_ERROR}",
                foreground="#c62828",
            )
            self.start_btn.config(state="disabled")
            return
        if not (DEMUCS_AVAILABLE or ASEP_AVAILABLE):
            self.env_label.config(
                text=(
                    "⚠ No hay motor de separación disponible.\n"
                    f"  Demucs: {DEMUCS_IMPORT_ERROR}\n"
                    f"  audio-separator: {ASEP_IMPORT_ERROR}"
                ),
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

        # GPU info + ffmpeg
        gpu_msg = ""
        if cuda_available():
            name = gpu_name() or "CUDA"
            gpu_msg = f"✓ GPU: {name} · "
        else:
            gpu_msg = "ℹ CPU only · "

        ffmpeg_msg = (
            f"ffmpeg OK ({Path(ffmpeg_dir).name}) · "
            if ffmpeg_dir
            else "⚠ ffmpeg NO encontrado (instalación necesaria) · "
        )

        if cuda_available():
            self.env_label.config(
                text=gpu_msg + ffmpeg_msg + "RoFormer será 20-100× más rápido que en CPU.",
                foreground="#2e7d32",
            )
        else:
            self.env_label.config(
                text=gpu_msg
                + ffmpeg_msg
                + "Recomendado: Demucs · 4 fuentes. RoFormer será muy lento.",
                foreground="#2e7d32" if ffmpeg_dir else "#ef6c00",
            )

        # Si falta ffmpeg y audio-separator está disponible, desactivamos RoFormer
        if ASEP_AVAILABLE and not ffmpeg_dir:
            messagebox.showwarning(
                "FFmpeg no encontrado",
                "No se encontró ffmpeg.exe en el sistema.\n\n"
                "El motor RoFormer / MDXC lo necesita para decodificar MP3, FLAC, etc.\n\n"
                "Instálalo con:  winget install -e --id Gyan.FFmpeg\n"
                "O agrega la carpeta que contiene ffmpeg.exe al PATH.\n\n"
                "Mientras tanto puedes usar los motores Demucs (4/6 fuentes).",
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
            messagebox.showwarning(
                "Sin archivo", "Carga primero un archivo de audio."
            )
            return

        # Chequeo de motor seleccionado
        mode = self.mode_var.get()
        if mode == self.MODE_ROFORMER and not ASEP_AVAILABLE:
            messagebox.showerror(
                "audio-separator no disponible",
                f"Motivo: {ASEP_IMPORT_ERROR}",
            )
            return
        if mode != self.MODE_ROFORMER and not DEMUCS_AVAILABLE:
            messagebox.showerror(
                "Demucs no disponible",
                f"Motivo: {DEMUCS_IMPORT_ERROR}",
            )
            return

        # Aviso si eligió RoFormer en CPU (mucho más lento)
        if mode == self.MODE_ROFORMER and not cuda_available():
            if not messagebox.askyesno(
                "RoFormer en CPU",
                "No se detectó GPU con CUDA y elegiste RoFormer / MDXC.\n\n"
                "En CPU, RoFormer es entre 20× y 100× más lento que en GPU. "
                "Una canción de 4 min puede tardar 30-90 min.\n\n"
                "¿Continuar de todas formas?\n\n"
                "(Recomendación: usa 'Demucs – 4 fuentes' en CPU.)",
            ):
                return

        self.is_processing = True
        self.start_btn.config(state="disabled")
        self.progress.start(15)
        self.status_label.config(text="Iniciando…", foreground="#1565c0")

        kwargs = {
            "input_path": self.input_file,
            "midside": self.midside_var.get(),
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
        self,
        input_path: str,
        engine: str,
        engine_model: str,
        midside: bool,
    ) -> None:
        try:
            track_stem = Path(input_path).stem
            # Limpiar la extensión del modelo (evita "...htdemucs_6s.yaml/bass.wav")
            model_tag = Path(engine_model).stem
            run_out = self.output_dir / f"{track_stem}__{model_tag}"
            run_out.mkdir(parents=True, exist_ok=True)

            if engine == "demucs":
                self._run_demucs(input_path, engine_model, run_out)
            elif engine == "asep":
                self._run_asep(input_path, engine_model, run_out)
            else:
                raise RuntimeError(f"Motor desconocido: {engine}")

            # Mid/Side opcional
            extra_files: list[Path] = []
            if midside:
                self.message_queue.put(
                    ("status", "Post-procesando pistas estéreo (Mid/Side/L/R)…")
                )
                extra_files = self._apply_midside(run_out)

            saved = sorted(p for p in run_out.iterdir() if p.suffix.lower() == ".wav")
            self.message_queue.put(("done", (str(run_out), saved)))

        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            self.message_queue.put(("error", f"{exc}\n\nDetalles:\n{tb}"))

    def _run_demucs(self, input_path: str, model_name: str, run_out: Path) -> None:
        device = "cuda" if cuda_available() else "cpu"
        self.message_queue.put((
            "status",
            f"[Demucs] Cargando '{model_name}' en {device}… "
            "(primera vez puede tardar algunos minutos para descargar pesos)",
        ))
        separator = DemucsSeparator(model=model_name, device=device)
        self.message_queue.put(("status", f"[Demucs] '{model_name}' separando…"))
        _origin, separated = separator.separate_audio_file(input_path)
        self.message_queue.put(("status", "[Demucs] Guardando archivos WAV…"))
        sample_rate = getattr(separator, "samplerate", 44100)
        self._save_dict_as_wav(separated, run_out, sample_rate)

    def _run_asep(self, input_path: str, model_filename: str, run_out: Path) -> None:
        # audio-separator escribe él mismo; usamos un tempdir y luego movemos.
        import tempfile

        with tempfile.TemporaryDirectory(prefix="asep_") as tmp:
            tmp_path = Path(tmp)
            sep = AudioSeparator(  # type: ignore[misc]
                output_dir=str(tmp_path),
                output_format="WAV",
            )
            dev = sep.setup_accelerated_inferencing_device()
            self.message_queue.put((
                "status",
                f"[RoFormer] Acelerador: {dev}  ·  Descargando / cargando "
                f"modelo '{model_filename}'…",
            ))
            sep.load_model(model_filename=model_filename)
            self.message_queue.put(("status", "[RoFormer] Ejecutando separación…"))
            out_files = sep.separate(input_path)  # tipo: list[str]

            moved: list[tuple[Path, Path]] = []
            for f in out_files:
                # ``audio-separator`` puede devolver paths absolutos o sólo el
                # basename; normalizamos siempre relativo a tmp_path.
                p = Path(f)
                src = p if p.is_absolute() else (tmp_path / p.name)
                if not src.exists():
                    # Tal vez está en un output_dir interno (subcarpeta model_t)
                    alt = tmp_path / p.name
                    if alt.exists():
                        src = alt
                    else:
                        self.message_queue.put((
                            "status",
                            f"[RoFormer] WARN: source no encontrado: {src}",
                        ))
                        continue
                new_name = self._clean_asep_name(src.name)
                if new_name is None:
                    continue
                dst = run_out / new_name
                if dst.exists():
                    dst.unlink()  # evita shutil.move fallando por copia previa
                shutil.move(str(src), str(dst))
                moved.append((src, dst))

            self.message_queue.put((
                "status",
                f"[RoFormer] Guardados {len(moved)} stem(s) en {run_out.name}",
            ))

    @staticmethod
    def _clean_asep_name(filename: str) -> Optional[str]:
        """Convierte 'song_(Vocals)_model_bs_roformer_xxx.wav' en 'vocals.wav'."""
        m = re.search(r"\(([^)]+)\)", filename)
        if not m:
            return None
        stem = m.group(1).strip().lower().replace(" ", "_")
        return f"{stem}.wav"

    def _save_dict_as_wav(self, separated: dict, folder: Path, sr: int) -> list[Path]:
        if sf is None:
            raise RuntimeError(
                "soundfile no está disponible para escribir WAV."
            )
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

    def _apply_midside(self, folder: Path) -> list[Path]:
        """Genera *_L / *_R / *_Mid / *_Side para guitar.wav y piano.wav."""
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
            stem = target[:-4]  # quita .wav
            for label, channel in (("L", left), ("R", right),
                                   ("Mid", mid), ("Side", side)):
                out = folder / f"{stem}_{label}.wav"
                # Soundfile espera (samples, channels)
                if label in ("L", "R", "Mid", "Side"):
                    arr = column_stack_2(channel)
                else:
                    arr = channel
                sf.write(str(out), arr, sr)
                produced.append(out)
        return produced

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
        names = ", ".join(Path(p).name for p in files[:8])
        more = "" if len(files) <= 8 else f"  (+{len(files) - 8} más)"
        self.status_label.config(
            text=f"✓ Separación completada — {len(files)} archivos: {names}{more}",
            foreground="#2e7d32", wraplength=760,
        )
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

    # --------------------------------------------- Abrir carpeta salida

    def _open_folder(self, folder: str) -> None:
        # Asegurar separadores estilo Windows
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
                "Carpeta de salida",
                f"No pude abrir la carpeta automáticamente.\n\n{path}",
            )


# Pequeña utilidad separada para evitar shadowing en el bucle Mid/Side
def column_stack_2(arr) -> "np.ndarray":  # type: ignore[name-defined]
    import numpy as np
    return np.column_stack((arr, arr))


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
