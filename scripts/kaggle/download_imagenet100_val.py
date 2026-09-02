from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

def _find_val_root(root: Path) -> Path | None:
    candidates = [root / "val.X", root]
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        class_dirs = [path for path in candidate.iterdir() if path.is_dir()]
        if len(class_dirs) == 100:
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download only ambityga/imagenet100 val.X from Kaggle."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("downloads/kaggle/imagenet100-val"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = _find_val_root(output_dir)
    if existing is not None and not args.force:
        print(f"[ready] validation root: {existing}")
        return

    archive = output_dir / "val.X.zip"
    if archive.is_file() and not args.force:
        print(f"[archive] using existing file: {archive}")
    else:
        try:
            import kagglehub
        except ImportError as error:
            raise RuntimeError(
                "kagglehub is required for API download; alternatively place "
                f"val.X.zip at {archive}"
            ) from error
        downloaded = Path(
            kagglehub.dataset_download(
                "ambityga/imagenet100",
                path="val.X.zip",
                force_download=args.force,
                output_dir=str(output_dir),
            )
        )
        print(f"[downloaded] {downloaded}")
        archive = downloaded if downloaded.is_file() else archive

    if archive.is_file() and zipfile.is_zipfile(archive):
        extract_tmp = output_dir / ".val_extracting"
        if extract_tmp.exists():
            shutil.rmtree(extract_tmp)
        extract_tmp.mkdir(parents=True)
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(extract_tmp)
        extracted = _find_val_root(extract_tmp)
        if extracted is None:
            raise RuntimeError(f"archive does not contain a 100-class val.X tree: {archive}")
        destination = output_dir / "val.X"
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(extracted), str(destination))
        shutil.rmtree(extract_tmp, ignore_errors=True)

    validation_root = _find_val_root(output_dir)
    if validation_root is None:
        raise RuntimeError(
            f"download completed, but no 100-class validation tree was found in {output_dir}"
        )
    image_suffixes = {".jpeg", ".jpg", ".png", ".webp"}
    image_count = sum(
        1
        for path in validation_root.rglob("*")
        if path.is_file() and path.suffix.lower() in image_suffixes
    )
    if image_count != 5000:
        raise RuntimeError(f"expected 5,000 validation images, found {image_count}")
    print(f"[ready] validation root: {validation_root}")
    print(f"[ready] images: {image_count:,}")


if __name__ == "__main__":
    main()
