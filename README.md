# 🎵 Separador Musical + 🎸 Transcripción de Guitarra (Tkinter)

Aplicación de escritorio (Tkinter) organizada en **3 pestañas** que **separa
fuentes de audio**, reproduce cada pista **en vivo dentro de la app**, mezcla
multitrack y **analiza guitarra** desde una misma ventana.

## 🖥️ Qué hace

**1. Separación de fuentes** con **3 motores**:

| # | Motor | Modelos | Stems |
|---|-------|---------|-------|
| 1 | **Demucs** | `htdemucs` | vocals · drums · bass · other |
| 2 | **Demucs** | `htdemucs_6s` | vocals · drums · bass · other · **guitar** · **piano** |
| 3 | **RoFormer / MDXC** | BS-RoFormer, DrumSep MDX23C, Demucs vía UVR (163 modelos disponibles) | según modelo elegido |

Y un **post-procesador estéreo** (Mid / Side / L / R) sobre `guitar.wav` o
`piano.wav`.

**Filtro Mid/Side de cancelación de fase (karaoke / limpiar público):** para
temas en vivo, donde el cantante está paneado al centro y el público en los
laterales:

* **Conservar centro (M = L+R)** — quita el público estéreo de la pista de voz.
* **Karaoke (L−R)** — quita la voz central y deja los laterales.
* **Atenuar público (M + 0.35·S)** — conserva la voz y baja el ambiente.

Se puede aplicar **automáticamente tras separar** (pestaña 1, con selector de
modo) o **por pista desde la pestaña 2** (botón 🎛️ en cada fila del mixer,
que reemplaza el WAV al guardar).

**3. 🎚️ Mezclador multitrack** — carga **automáticamente** los stems separados
con botones **▶ ⏸ ⏹** que suenan **dentro del propio programa**, **mute (M)**,
**solo (S)** y **volumen** por pista con un **master global**, y permite
generar/descargar una mezcla WAV respetando esos ajustes.

**2. 🎸 Análisis y Transcripción de Guitarra** (pestaña 3): sobre un WAV de
guitarra (p. ej. `guitar.wav` de la separación, que se carga automáticamente):

* **Afinación estimada** — detecta la cuerda grave (f0 con `librosa`) y
  la compara con afinaciones conocidas (Estándar, Drop D, medio tono…).
* **Línea de tiempo de acordes** — croma + plantillas; se muestra en texto
  y como figura en una ventana aparte.
* **Audio → MIDI** — transcripción con **Basic Pitch** (Spotify), archivo
  `.mid` compatible con TuxGuitar / Guitar Pro.

**Prioridad GPU, fallback CPU en todo el proceso:**

* **Separación (Demucs / RoFormer)** → GPU CUDA si está disponible, si no CPU.
* **Transcripción Audio → MIDI (Basic Pitch)** → GPU CUDA: el código precarga
  las DLLs de CUDA de PyTorch (`onnxruntime.preload_dlls()`) y usa
  `onnxruntime-gpu` compilado contra tu mismo CUDA/cuDNN. Cae a CPU si no hay
  GPU compatible.
* **Afinación y acordes (librosa)** → corren en CPU (rapidísimos; librosa no
  acelera por GPU), en paralelo sin bloquear la separación.

La separación corre en un **hilo secundario** con soporte automático de
**GPU CUDA**, incluyendo **NVIDIA RTX 5070 Laptop** (Blackwell, sm_120):
**separación y transcripción MIDI usan la GPU; afinación/acordes son CPU
(rápidos).** Si no hay GPU, todo cae a CPU automáticamente.

---

## 📁 Estructura

```
main.py              App de escritorio por pestañas (entrada principal)
playback.py          Motor de reproducción de audio en vivo (sounddevice)
guitar_analysis.py   Afinación, acordes y transcripción Audio → MIDI
midside.py           Filtro Mid/Side de cancelación de fase (karaoke / público)
ffmpeg_utils.py      Detección de FFmpeg y GPU CUDA
separation.py        Utilidades de separación/mix (opcional)
check_env.py         Diagnóstico del entorno
```

---

## ⚠️ Requisitos importantes

* **Python 3.12** (recomendado) — las dependencias aún no tienen wheels estables para 3.14.
* **FFmpeg** en el PATH del sistema (audio-separator lo usa).
* (Opcional) **GPU NVIDIA** con la build CUDA correcta de PyTorch.

---

## 🚀 Instalación

### 1. Python 3.12 + venv

```bash
py -3.12 -m venv .venv          # Windows (3.12 debe estar instalado)
# o:
python3.12 -m venv .venv        # Linux / macOS

# Activar
.venv\Scripts\activate           # Windows
source .venv/bin/activate        # Linux / macOS
```

### 2. PyTorch con soporte GPU (RTX 5070 / Blackwell)

```bash
# Blackwell / CUDA 12.8
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

> Si tienes otra GPU NVIDIA, elige la build CUDA correspondiente en
> https://pytorch.org/get-started/locally/

### 3. Resto de las dependencias

```bash
pip install -r requirements.txt
```

### 4. FFmpeg (audio-separator lo necesita)

```bash
# Windows
winget install -e --id Gyan.FFmpeg

# macOS
brew install ffmpeg

# Debian/Ubuntu
sudo apt-get install ffmpeg
```

---

## ▶️ Uso (Escritorio)

```bash
run.bat        # Windows — recomendado: usa el venv con CUDA automáticamente
```

o manualmente:

```bash
.venv\Scripts\activate   # Windows
python main.py
```

> ⚠️ En Windows, si escribes simplemente `python main.py` desde una terminal
> que usa el Python global (p. ej. 3.14 con torch **+cpu**), verás "CPU only"
> aunque tengas RTX. Usa `run.bat` para garantizar el venv con CUDA.

1. **Cargar archivo** (MP3/WAV/FLAC/OGG/M4A).
2. Elegir **motor**:
   * *Demucs — 4 fuentes* → separación clásica rápida
   * *Demucs — 6 fuentes* → añade guitarra + piano
   * *RoFormer / MDXC* → modelos UVR de mejor calidad, GPU recomendado
3. (RoFormer) Elegir **modelo** entre la lista curada.
4. (Opcional) Marcar **post-procesamiento estéreo**.
5. **Empezar separación**.
6. Al terminar se abre la carpeta con los `.wav`.

Al terminar, la app **pasa automáticamente a la pestaña 2 · Multitrack** y
carga los stems. Ahí puedes reproducir cada pista **en vivo dentro del
programa** (▶⏸⏹), silenciar (M), solo (S), ajustar volumen y usar el
**master** global. Para escuchar la combinación de ajustes pulsa
**🎧 Generar y Reproducir Mezcla** (suena también dentro de la app) y
**⬇ Descargar mezcla (.wav)** para guardarla.

En la **pestaña 3 · Análisis y Transcripción**:

7. Se carga automáticamente `guitar.wav` (o `bass.wav` si no hay guitarra) al
   separar; también puedes pulsar **🎸 Cargar Guitarra** y elegir otro WAV.
8. **🔍 Analizar Guitarra** → muestra afinación y línea de tiempo de acordes
   (la figura sale en una ventana aparte).
9. **🎹 Generar MIDI** → guarda `transcripcion_guitarra.mid` junto al audio
   abierto (compatible con TuxGuitar / Guitar Pro).

### Salida

Por defecto se guarda en la **carpeta Música de Windows** (respeta la
redirección a OneDrive), dentro de una carpeta con el nombre del programa y
otra con el nombre del tema:

```
C:\Users\tu_usuario\Music\Separador_Musical\<canción>\
   ├── vocals.wav
   ├── instrumental.wav
   ├── ...
```

---

> ⚠️ **Python 3.12 / Windows**: el análisis Audio→MIDI usa `basic-pitch`, que
> no se puede instalar con `pip install -r requirements.txt` (exige
> `tensorflow<2.15.1`, sin wheels para cp312). Instálalo aparte con el modelo
> ONNX incluido, sin TensorFlow:
>
> ```bash
> pip install basic-pitch --no-deps
> pip install onnxruntime pretty-midi mir-eval resampy scikit-learn
> ```

---

## 🧠 ¿Cómo funciona?

* **3 motores** usan la librería que prefieras (Demucs o audio-separator).
* **Filtro Mid/Side**: la transformación `Mid = (L+R)/2` y `Side = (L−R)/2`
  separa lo central (voz) de lo lateral (público/ambiente). Según el modo
  elegido se mezclan con distintas ganancias y se normaliza el pico para
  evitar clipping.
* El **análisis de guitarra** usa `librosa` (afinación/acordes) y Basic Pitch
  (MIDI) de forma **bloqueante junto a la GUI**: los botones se deshabilitan
  mientras corre y vuelven a activarse al terminar.
* Toda la inferencia corre en `threading.Thread(daemon=True)`.
* La **reproducción en vivo** usa `sounddevice` con un **stream continuo
  (callback de PortAudio)** y carga el archivo en un hilo aparte, así
  pausa/continúa/volumen responden al instante, el audio fluye sin cortes y
  la GUI nunca se congela; si `sounddevice` no está disponible, las pistas
  se abren en el reproductor del sistema.
* El progreso se reporta al hilo principal vía `queue.Queue` y se lee
  con `root.after(120, ...)`.
* **CUDA** se detecta con `torch.cuda.is_available()`. Si no hay GPU,
  se usa CPU (más lento).
* **audio-separator** descarga los modelos de HuggingFace la primera vez
  (~150-300 MB por modelo). Se cachea en `%TEMP%\audio-separator-models\`.
* **Post-procesador Mid/Side**: para `guitar.wav` o `piano.wav` estéreo,
  genera:
    * `guitar_L.wav`  — sólo canal izquierdo
    * `guitar_R.wav`  — sólo canal derecho
    * `guitar_Mid.wav` — `(L + R) / 2`
    * `guitar_Side.wav` — `(L - R) / 2`

---

## 🩹 Diagnóstico

```bash
python check_env.py
```

Te indica exactamente qué paquetes están instalados, dónde, y si CUDA,
FFmpeg y `onnxruntime-gpu` están listos.

---

## ⚡ CPU vs GPU — qué esperar

La app **funciona automáticamente en CPU o GPU**. Detecta CUDA y elige el
mejor dispositivo. La barra inferior avisa siempre del modo activo.

| Motor | CPU (4 min canción) | GPU (RTX 5070) |
|-------|---------------------|----------------|
| **Demucs · 4 fuentes** | 8–20 min | ~30 s |
| **Demucs · 6 fuentes** | 12–30 min | ~50 s |
| **RoFormer · BS-RoFormer 317** | **30–90 min** | **~75 s** |
| **MDX23C DrumSep** | 15–40 min | ~60 s |

> Si ves la barra inferior en **naranja** (no detecta GPU), la app te
> mostrará un aviso al pulsar *Empezar* con RoFormer/MDXC, recomendando
> cambiar a Demucs.

### Instalar PyTorch CUDA vs CPU

Si tu máquina **tiene GPU NVIDIA**:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Si **NO tiene GPU** (o no quieres instalar drivers NVIDIA):

```bash
pip install torch torchaudio     # build CPU (más pequeña, ~200 MB)
```

El resto de las dependencias (`pip install -r requirements.txt`) son
iguales. **No hace falta** `audio-separator[gpu]` si vas a usar sólo CPU;
instalar `audio-separator` a secas ya basta, pero `[gpu]` no molesta.

---

## 🩹 Solución de problemas

| Problema | Solución |
|----------|----------|
| "PyTorch no instalado" | `pip install torch torchaudio` |
| "Demucs no instalado" | `pip install "demucs>=4.0.1"` |
| "audio-separator no instalado" | `pip install "audio-separator[gpu]>=0.44"` |
| FFmpeg is not installed | `winget install -e --id Gyan.FFmpeg` |
| CUDA disponible pero no se usa | Comprueba que PyTorch sea la build con CUDA: `pip show torch` debe decir `+cu128` o `+cu126` |
| La app va muy lenta | O usa GPU (instalando build CUDA correcta), o elige Demucs 4 fuentes en lugar de RoFormer |
| No detecta GPU pero sí tengo NVIDIA | Instala la build CUDA correcta con `--index-url https://download.pytorch.org/whl/cu128` |
| No se escucha audio en la app | `sounddevice` necesita PortAudio (viene con el wheel). Verifica que el dispositivo de salida esté activo; sin él las pistas se abren en el reproductor del sistema |

---

## ⚠️ Limitación importante: separación de guitarras

En la familia de modelos **UVR/Demucs RoFormer** que `audio-separator`
proporciona, **ningún modelo** separa explícitamente **guitarra
eléctrica de acústica**: el modelo multi-stem de mejor calidad
(`htdemucs_6s`) sigue agrupando todas las guitarras en un único
`guitar.wav`.

Lo que sí puedes hacer en esta app para acercarte a esa separación:

1. **Usar el post-procesador estéreo Mid/Side / L/R** sobre
   `guitar.wav` — útil cuando hay guitarras paneadas a lados
   distintos (caso típico en rock/metal/pop).
2. Para separaciones más finas (eléctrica vs acústica como tales),
   necesitas servicios comerciales como
   [Moises.ai](https://moises.ai) o [LALAL.AI](https://www.lalal.ai)
   que tienen modelos propietarios entrenados con esas etiquetas.

---

## 📜 Licencia

Demucs (MIT) · audio-separator (MIT) · UVR models (MIT, entrenados por la comunidad).
Esta interfaz es un envoltorio independiente sin garantías.
