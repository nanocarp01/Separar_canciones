# 🎵 Demucs · RoFormer — Separador de Fuentes de Audio (Tkinter)

Aplicación de escritorio escrita en Python + Tkinter que separa las fuentes de
audio (voces, batería, bajo, guitarra, piano, etc.) de cualquier archivo
MP3, WAV, FLAC u OGG. Incluye **3 motores**:

| # | Motor | Modelos | Stems |
|---|-------|---------|-------|
| 1 | **Demucs** | `htdemucs` | vocals · drums · bass · other |
| 2 | **Demucs** | `htdemucs_6s` | vocals · drums · bass · other · **guitar** · **piano** |
| 3 | **RoFormer / MDXC** | BS-RoFormer, DrumSep MDX23C, Demucs vía UVR (163 modelos disponibles) | según modelo elegido |

Y un **post-procesador estéreo** (Mid / Side / L / R) sobre `guitar.wav` o
`piano.wav` para cuando hay guitarras paneadas a lados distintos.

La separación corre en un **hilo secundario** con soporte automático de
**GPU CUDA**, incluyendo **NVIDIA RTX 5070 Laptop** (Blackwell, sm_120).

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

## ▶️ Uso

```bash
python main.py
```

1. **Cargar archivo** (MP3/WAV/FLAC/OGG/M4A).
2. Elegir **motor**:
   * *Demucs — 4 fuentes* → separación clásica rápida
   * *Demucs — 6 fuentes* → añade guitarra + piano
   * *RoFormer / MDXC* → modelos UVR de mejor calidad, GPU recomendado
3. (RoFormer) Elegir **modelo** entre la lista curada.
4. (Opcional) Marcar **post-procesamiento estéreo**.
5. **Empezar separación**.
6. Al terminar se abre la carpeta con los `.wav`.

### Salida

Por defecto: `~/DemucsSeparated/<canción>__<modelo>/`, p.ej.:

```
C:\Users\tu_usuario\DemucsSeparated\cancion__model_bs_roformer_ep_317_sdr_12\
   ├── vocals.wav
   ├── instrumental.wav
   ├── ...
```

---

## 🧠 ¿Cómo funciona?

* **3 motores** usan la librería que prefieras (Demucs o audio-separator).
* Toda la inferencia corre en `threading.Thread(daemon=True)`.
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
