"""
Motor de reproducción de audio en vivo para la GUI (pestaña Multitrack).

Usa ``sounddevice`` (PortAudio) con un **stream continuo basado en callback**:
el audio se carga en un hilo aparte (la GUI nunca se congela) y el flujo sale
de forma ininterrumpida, sin cortes entre bloques. El volumen por pista, el
master y pausa/continuar se aplican al instante.

Modo de uso:
    eng = PlaybackEngine()
    eng.play(path, volume=1.0, master=1.0)  # empieza a sonar PATH (carga async)
    eng.pause()                              # pausa temporal (conserva posición)
    eng.resume()                             # continúa donde quedó
    eng.stop()                               # detiene y vuelve al inicio
    eng.set_volume(0.8)                      # volumen de la pista (en vivo)
    eng.set_master(1.0)                      # volumen global (en vivo)
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger("Playback")

try:
    import sounddevice as sd

    SOUNDDEVICE_AVAILABLE = True
    SOUNDDEVICE_IMPORT_ERROR = ""
except Exception as exc:  # noqa: BLE001
    sd = None  # type: ignore[assignment]
    SOUNDDEVICE_AVAILABLE = False
    SOUNDDEVICE_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


class PlaybackEngine:
    """Reproduce un WAV en vivo con play/pausa/detener y volumen en tiempo real."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: Optional[np.ndarray] = None
        self._sr: int = 22050
        self._volume: float = 1.0
        self._master: float = 1.0
        self._pos: int = 0          # frame actual de reproducción
        self._paused: bool = False
        self._finished: bool = True
        self._stream = None         # sd.OutputStream activo (o None)
        self.last_error: Optional[str] = None

    # ---------------------------------------------------------- consultas
    def is_playing(self) -> bool:
        with self._lock:
            return self._data is not None and not self._finished and not self._paused

    def is_paused(self) -> bool:
        with self._lock:
            return self._data is not None and not self._finished and self._paused

    def position_seconds(self) -> float:
        with self._lock:
            if self._data is None:
                return 0.0
            return self._pos / self._sr

    def duration_seconds(self) -> float:
        with self._lock:
            if self._data is None:
                return 0.0
            return len(self._data) / self._sr

    # ---------------------------------------------------------- controles
    def play(self, path: str, volume: Optional[float] = None,
             master: Optional[float] = None) -> None:
        """Empieza a reproducir *path*. La carga corre en un hilo: no bloquea."""
        if not SOUNDDEVICE_AVAILABLE:
            raise RuntimeError(
                "sounddevice no está disponible para reproducir en vivo: "
                f"{SOUNDDEVICE_IMPORT_ERROR}"
            )
        self.last_error = None
        threading.Thread(
            target=self._load_and_start,
            args=(path, volume, master),
            daemon=True,
            name="playback-load",
        ).start()

    def pause(self) -> None:
        """Pausa conservando la posición (el stream sigue abierto, sale silencio)."""
        with self._lock:
            if self._data is not None and not self._finished and not self._paused:
                self._paused = True

    def resume(self) -> None:
        with self._lock:
            if self._data is not None and not self._finished and self._paused:
                self._paused = False

    def stop(self) -> None:
        """Detiene, cierra el stream y vuelve al inicio."""
        with self._lock:
            self._close_stream_locked()
            self._pos = 0
            self._paused = False
            self._finished = True
            self._data = None

    def set_volume(self, value: float) -> None:
        with self._lock:
            self._volume = float(value)

    def set_master(self, value: float) -> None:
        with self._lock:
            self._master = float(value)

    # ------------------------------------------------------------ interno
    def _load_and_start(self, path: str, volume, master) -> None:
        """Lee el archivo y abre el stream continuo (hilo aparte)."""
        try:
            import soundfile as sf

            data, sr = sf.read(path, always_2d=True, dtype="float32")
            if data.ndim == 1:
                data = data[:, None]
            data = np.ascontiguousarray(data, dtype=np.float32)

            with self._lock:
                self._close_stream_locked()
                self._data = data
                self._sr = int(sr)
                if volume is not None:
                    self._volume = float(volume)
                if master is not None:
                    self._master = float(master)
                self._pos = 0
                self._paused = False
                self._finished = False
            self._open_stream()
        except Exception as exc:  # noqa: BLE001
            logger.warning("No se pudo reproducir %s: %s", path, exc)
            self.last_error = str(exc)
            with self._lock:
                self._finished = True
                self._data = None

    def _open_stream(self) -> None:
        with self._lock:
            data = self._data
            sr = self._sr
            if data is None:
                return
            channels = int(data.shape[1])
        try:
            stream = sd.OutputStream(
                samplerate=sr,
                channels=channels,
                dtype="float32",
                callback=self._callback,
                blocksize=0,
            )
            if not stream.active:
                stream.start()
        except Exception as exc:  # noqa: BLE001
            logger.warning("No se pudo abrir el stream de audio: %s", exc)
            self.last_error = str(exc)
            with self._lock:
                self._finished = True
            return
        with self._lock:
            self._stream = stream

    def _close_stream_locked(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:  # noqa: BLE001
                pass

    def _callback(self, outdata, frames, _time_info, _status) -> None:
        """Rellena el buffer de salida con el tramo actual (hilo de audio)."""
        with self._lock:
            if self._data is None or self._finished or self._paused:
                outdata.fill(0.0)
                return
            data = self._data
            pos = self._pos
            vol = self._volume * self._master
            end = pos + frames
            if end >= len(data):
                n = len(data) - pos
                if n > 0:
                    outdata[:n] = data[pos:] * vol
                if n < frames:
                    outdata[n:] = 0.0
                self._pos = len(data)
                self._finished = True
            else:
                outdata[:] = data[pos:end] * vol
                self._pos = end
