"""
Manifiesto de modelos de separación: procedencia, hash SHA-256, licencia y capacidades.

Este módulo lee ``model_manifest.json`` (en la misma carpeta) y verifica que los
modelos que usa la app estén registrados, con su hash SHA-256 y su licencia.

Filosofía (prompt "Limbus Split Pro"):
  * Modo **strict**  -> la app BLOQUEA cualquier modelo no registrado en el
                        manifiesto (compilación distribuible / pública).
  * Modo **desarrollo** -> solo avisa al elegir un modelo no auditado.

Uso desde terminal:

    python model_manifest.py --check              # estado de todos los modelos
    python model_manifest.py --register <archivo> # calcula hash y genera el JSON
    python model_manifest.py --verify <archivo>   # hash + ¿está registrado?
    python model_manifest.py --check --strict     # sale con código 1 si algo falla (CI)

Dónde busca los modelos (por orden):
  * Torch hub checkpoints  (TORCH_HOME/HUB_CACHE o ~/.cache/torch/hub/checkpoints)
  * audio-separator        (AUDIO_SEPARATOR_MODEL_DIR, ~/.audio-separator-models,
                            %TEMP%\\audio-separator-models, %LOCALAPPDATA%\\audio-separator-models)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

# En Windows la consola va por cp1252 y rompe con caracteres no ASCII.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

MANIFEST_FILE = Path(__file__).resolve().parent / "model_manifest.json"

_CHUNK = 1 << 20  # 1 MiB


# ---------------------------------------------------------------------------
# Rutas de búsqueda
# ---------------------------------------------------------------------------
def _candidate_dirs() -> list[Path]:
    dirs: list[Path] = []
    seen: set[str] = set()

    def add(p: Path | None) -> None:
        if p is None:
            return
        key = str(p).lower()
        if key not in seen:
            seen.add(key)
            dirs.append(p)

    home = Path.home()

    # Torch hub checkpoints (Demucs)
    torch_home = os.environ.get("TORCH_HOME") or os.environ.get("HUB_CACHE")
    if torch_home:
        add(Path(torch_home) / "checkpoints")
    add(home / ".cache" / "torch" / "hub" / "checkpoints")

    # audio-separator (RoFormer / MDX / UVR)
    asep = os.environ.get("AUDIO_SEPARATOR_MODEL_DIR")
    if asep:
        add(Path(asep))
    add(home / ".audio-separator-models")
    for env in ("TEMP", "TMP"):
        tmp = os.environ.get(env)
        if tmp:
            add(Path(tmp) / "audio-separator-models")
    localapp = os.environ.get("LOCALAPPDATA")
    if localapp:
        add(Path(localapp) / "audio-separator-models")
    # audio-separator usa por defecto "/tmp/audio-separator-models/":
    # en Windows Python eso es C:\tmp\audio-separator-models; en POSIX /tmp/…
    add(Path("/tmp") / "audio-separator-models")
    return dirs


def find_file(filename: str, dirs: list[Path] | None = None) -> Path | None:
    """Busca ``filename`` en los directorios de modelos (incluye 1 subnivel)."""
    for d in (dirs if dirs is not None else _candidate_dirs()):
        if not d.is_dir():
            continue
        cand = d / filename
        if cand.is_file():
            return cand
        try:
            for sub in d.iterdir():
                if sub.is_dir():
                    inner = sub / filename
                    if inner.is_file():
                        return inner
        except (PermissionError, OSError):
            continue
    return None


# ---------------------------------------------------------------------------
# Carga / consulta
# ---------------------------------------------------------------------------
def load_manifest(path: str | Path | None = None) -> dict | None:
    p = Path(path) if path else MANIFEST_FILE
    if not p.exists():
        return None
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return None


def manifest_mode(manifest: dict | None = None) -> str:
    """'strict' bloquea modelos no registrados; 'dev' solo avisa."""
    m = manifest if manifest is not None else load_manifest()
    if m is None:
        return "dev"
    return "strict" if m.get("strict") else "dev"


def manifest_path() -> Path:
    return MANIFEST_FILE


def sha256_file(path: str | Path, chunk: int = _CHUNK) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _entry_matches(entry: dict, filename: str) -> bool:
    low = filename.lower()
    for f in entry.get("files", []):
        if str(f.get("filename", "")).lower() == low:
            return True
    for cfg in entry.get("configs", []):
        if str(cfg).lower() == low:
            return True
    if str(entry.get("engine_name", "")).lower() == low:
        return True
    return False


def is_registered(filename: str, manifest: dict | None = None) -> dict | None:
    """Devuelve la entrada del manifiesto que cubre ``filename`` o None."""
    m = manifest if manifest is not None else load_manifest()
    if m is None:
        return None
    for entry in m.get("models", []):
        if _entry_matches(entry, filename):
            return entry
    return None


# ---------------------------------------------------------------------------
# Verificación
# ---------------------------------------------------------------------------
def verify_manifest(manifest: dict | None = None,
                    dirs: list[Path] | None = None) -> list[dict]:
    """Verifica cada entrada del manifiesto contra el disco.

    Devuelve una lista de informes por modelo con el estado por archivo:
      * hash_mismatch  -> el hash real NO coincide con el registrado (grave)
      * missing        -> ningún archivo del modelo encontrado
      * partial        -> algunos archivos encontrados y otros no
      * hash_pending   -> encontrado pero sin SHA-256 registrado en el manifiesto
      * ok             -> encontrado y con hash coincidente
    """
    m = manifest if manifest is not None else load_manifest()
    if m is None:
        return [{
            "id": "<sin manifiesto>", "name": "", "license": "?",
            "commercial": "?", "redistribution": "?",
            "status": "missing", "message": f"No existe {MANIFEST_FILE}", "files": [],
        }]

    out: list[dict] = []
    for entry in m.get("models", []):
        files: list[dict] = []
        for f in entry.get("files", []):
            fn = str(f.get("filename", ""))
            path = find_file(fn, dirs)
            rec = {
                "filename": fn,
                "found": path is not None,
                "path": str(path) if path else None,
                "expected_sha256": f.get("sha256"),
                "expected_size": f.get("size"),
                "actual_sha256": None,
                "actual_size": None,
                "hash_ok": None,
                "hash_pending": False,
                "hash_mismatch": False,
                "missing": path is None,
            }
            if path is not None:
                rec["actual_size"] = path.stat().st_size
                expected = f.get("sha256")
                if expected:
                    actual = sha256_file(path)
                    rec["actual_sha256"] = actual
                    rec["hash_ok"] = actual == expected
                    rec["hash_mismatch"] = actual != expected
                else:
                    rec["hash_pending"] = True
            files.append(rec)

        any_mismatch = any(r["hash_mismatch"] for r in files)
        any_found = any(r["found"] for r in files)
        any_missing = any(r["missing"] for r in files)
        any_pending = any(r["hash_pending"] for r in files)
        if any_mismatch:
            status = "hash_mismatch"
        elif any_missing and not any_found:
            status = "missing"
        elif any_missing:
            status = "partial"
        elif any_pending:
            status = "hash_pending"
        else:
            status = "ok"

        out.append({
            "id": entry.get("id", "?"),
            "name": entry.get("name", ""),
            "license": entry.get("license", "?"),
            "commercial": entry.get("commercial", "?"),
            "redistribution": entry.get("redistribution", "?"),
            "status": status,
            "message": entry.get("note", ""),
            "files": files,
        })
    return out


_STATUS_TAG = {
    "ok": "✓ OK",
    "hash_pending": "⚠ SIN HASH",
    "missing": "✗ NO ENCONTRADO",
    "partial": "◐ PARCIAL",
    "hash_mismatch": "✗ HASH NO COINCIDE",
}


def format_verify(entries: list[dict]) -> str:
    lines: list[str] = []
    for e in entries:
        lines.append(f"[{_STATUS_TAG.get(e['status'], '?')}] {e['id']}  (licencia: {e['license']})")
        for r in e["files"]:
            if r["missing"]:
                lines.append(f"        ✗ falta: {r['filename']}")
            elif r["hash_pending"]:
                lines.append(f"        ! encontrado, sin hash registrado: {r['filename']}  (usa --register)")
            elif r["hash_mismatch"]:
                lines.append(f"        ✗ HASH NO COINCIDE: {r['filename']}")
            else:
                short = (r["actual_sha256"] or "?")[:16]
                lines.append(f"        ✓ {r['filename']}  sha256 {short}…")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Registro de un modelo nuevo
# ---------------------------------------------------------------------------
def register_model_file(path: str | Path, filename: str | None = None,
                        entry_id: str | None = None) -> dict:
    """Calcula el SHA-256 de un archivo y devuelve el bloque JSON para el manifiesto."""
    p = Path(path)
    digest = sha256_file(p)
    return {
        "id": entry_id or f"local.{p.name}",
        "filename": filename or p.name,
        "sha256": digest,
        "size": p.stat().st_size,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Verificador del manifiesto de modelos (SHA-256 y licencias).",
    )
    ap.add_argument("--check", action="store_true",
                    help="Verifica el estado de todos los modelos del manifiesto (por defecto).")
    ap.add_argument("--register", metavar="ARCHIVO",
                    help="Calcula el SHA-256 de un modelo y muestra el JSON para el manifiesto.")
    ap.add_argument("--register-file", metavar="NOMBRE",
                    help="Nombre canónico del archivo al registrarlo (por defecto: el basename).")
    ap.add_argument("--verify", metavar="ARCHIVO",
                    help="Comprueba si un archivo concreto está registrado y si su hash coincide.")
    ap.add_argument("--strict", action="store_true",
                    help="Modo CI: sale con código 1 si algún modelo falta o su hash no coincide.")
    args = ap.parse_args(argv)

    if args.register:
        block = register_model_file(args.register, args.register_file)
        print("JSON para añadir al manifiesto (en el array 'models' o en el campo 'files' "
              "de la entrada correspondiente):")
        print(json.dumps(block, ensure_ascii=False, indent=2))
        return 0

    if args.verify:
        p = Path(args.verify)
        if not p.exists():
            print(f"✗ No existe: {p}")
            return 2
        digest = sha256_file(p)
        print(f"sha256  {digest}")
        print(f"size    {p.stat().st_size}")
        entry = is_registered(p.name)
        if entry is None:
            print(f"✗ '{p.name}' NO está registrado en el manifiesto. Regístralo con --register.")
            return 1
        expected = None
        for f in entry.get("files", []):
            if str(f.get("filename", "")).lower() == p.name.lower():
                expected = f.get("sha256")
        if expected is None:
            print(f"◐ '{p.name}' está en el manifiesto ({entry['id']}) pero sin hash registrado.")
            return 0
        if digest == expected:
            print(f"✓ HASH OK  ({entry['id']})")
            return 0
        print(f"✗ HASH NO COINCIDE  ({entry['id']})")
        return 1

    entries = verify_manifest()
    mode = manifest_mode()
    mode_desc = ('ESTRICTO (bloquea modelos no auditados)' if mode == 'strict'
                 else 'DESARROLLO (NO estricto — solo avisa)')
    print(f"Manifiesto: {MANIFEST_FILE}")
    print(f"Modo: {mode_desc}")
    print()
    print(format_verify(entries))
    print()
    bad = [e for e in entries if e["status"] in ("missing", "partial", "hash_mismatch")]
    if bad:
        print(f"Resumen: {len(entries)} modelo(s) → {len(bad)} con problemas.")
        if args.strict or mode == "strict":
            print("Modo estricto: FALLO.")
            return 1
    else:
        print(f"Resumen: {len(entries)} modelo(s) verificados.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
