from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


WEIGHT_SUFFIXES = {
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".bin",
}


def package_run_artifacts(runs_root: Path, output: Path) -> tuple[int, int, int]:
    """Zip all run artifacts except model/checkpoint weights."""
    runs_root = runs_root.resolve()
    output = output.resolve()
    if not runs_root.is_dir():
        raise FileNotFoundError(f"runs root does not exist: {runs_root}")
    output.parent.mkdir(parents=True, exist_ok=True)

    included_files = 0
    included_bytes = 0
    excluded_weights = 0
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(runs_root.rglob("*")):
            if not path.is_file() or path.resolve() == output:
                continue
            if path.suffix.lower() in WEIGHT_SUFFIXES or path.name.endswith(".pt.tmp"):
                excluded_weights += 1
                continue
            archive.write(path, Path("runs") / path.relative_to(runs_root))
            included_files += 1
            included_bytes += path.stat().st_size
    return included_files, included_bytes, excluded_weights


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package Kaggle run metrics and diagnostics without checkpoints."
    )
    parser.add_argument("--runs-root", type=Path, default=Path("/kaggle/working/runs"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/kaggle/working/intp_run_artifacts_no_weights.zip"),
    )
    args = parser.parse_args()
    files, size, excluded = package_run_artifacts(args.runs_root, args.output)
    print(
        f"[package] {args.output} | {files} files | {size / 1024:.1f} KiB input"
        f" | excluded {excluded} weight files",
        flush=True,
    )


if __name__ == "__main__":
    main()
