"""Build a clean Zenodo-ready archive of the artifact."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "dist" / "error-bounds-artifact.zip"
EXCLUDED_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    "dist",
    "results",
    "work",
    ".pytest_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".orig"}


def include(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if any(part in EXCLUDED_PARTS for part in relative.parts):
        return False
    if path.suffix in EXCLUDED_SUFFIXES:
        return False
    if relative.parts[0] == "notebooks" or relative.parts[:2] == ("reference_results", "figures"):
        return False
    if path.name == ".DS_Store" or path.is_symlink():
        return False
    return True


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a clean artifact archive.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.suffix != ".zip" and not output.name.endswith(".tar.gz"):
        parser.error("--output must end in .zip or .tar.gz")
    output.parent.mkdir(parents=True, exist_ok=True)

    files = []
    excluded_files = {output, Path(str(output) + ".sha256"), ROOT / "SHA256SUMS"}
    for directory, directories, names in os.walk(ROOT):
        directories[:] = [name for name in directories if include(Path(directory) / name)]
        files.extend(Path(directory) / name for name in names
                     if include(Path(directory) / name)
                     and (Path(directory) / name).resolve() not in excluded_files)
    files.sort()
    if not files:
        raise RuntimeError("No artifact files found.")
    manifest = ROOT / "SHA256SUMS"
    manifest.write_text("".join(f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}\n"
                                for path in files), encoding="utf-8")
    files.append(manifest)
    if output.suffix == ".zip":
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in files:
                archive.write(path, arcname=Path("error-bounds-artifact") / path.relative_to(ROOT))
    else:
        with tarfile.open(output, "w:gz", compresslevel=9) as archive:
            for path in files:
                archive.add(path, arcname=Path("error-bounds-artifact") / path.relative_to(ROOT), recursive=False)

    checksum = sha256(output)
    checksum_path = output.with_suffix(output.suffix + ".sha256")
    checksum_path.write_text(
        f"{checksum}  {output.name}\n",
        encoding="ascii",
    )
    print(f"Archive: {output}")
    print(f"Files: {len(files)}")
    print(f"Size: {output.stat().st_size:,} bytes")
    print(f"SHA-256: {checksum}")
    print(f"Checksum file: {checksum_path}")


if __name__ == "__main__":
    main()
