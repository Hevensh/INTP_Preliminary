from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import zipfile
from pathlib import Path
from typing import Any


DATASET_OWNER = "ambityga"
DATASET_SLUG = "imagenet100"
DATASET_VERSION = 8
VALIDATION_PREFIX = "val.X"


def _find_val_root(root: Path) -> Path | None:
    candidates = [root / "val.X", root]
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        class_dirs = [path for path in candidate.iterdir() if path.is_dir()]
        if len(class_dirs) == 100:
            return candidate
    return None


def _list_tree(api_client: Any, path: str) -> tuple[list[str], list[str]]:
    from kagglesdk.datasets.types.dataset_api_service import (
        ApiListTreeDatasetFilesRequest,
    )

    directories: list[str] = []
    files: list[str] = []
    page_token = ""
    while True:
        request = ApiListTreeDatasetFilesRequest()
        request.owner_slug = DATASET_OWNER
        request.dataset_slug = DATASET_SLUG
        request.dataset_version_number = DATASET_VERSION
        request.path = path
        request.page_size = 1000
        if page_token:
            request.page_token = page_token
        response = api_client.datasets.dataset_api_client.list_tree_dataset_files(
            request
        )
        directories.extend(item.name for item in response.directories)
        files.extend(item.name for item in response.files)
        page_token = response.next_page_token
        if not page_token:
            return directories, files


def _download_validation_tree(output_dir: Path, *, workers: int, force: bool) -> None:
    try:
        from kagglehub.clients import build_kaggle_client, download_file
        from kagglehub.handle import DatasetHandle
        from kagglesdk.datasets.types.dataset_api_service import (
            ApiDownloadDatasetRequest,
        )
    except ImportError as error:
        raise RuntimeError(
            "kagglehub is required for API download; install it or place a "
            f"prepared val.X.zip at {output_dir / 'val.X.zip'}"
        ) from error

    handle = DatasetHandle(DATASET_OWNER, DATASET_SLUG, DATASET_VERSION)
    with build_kaggle_client() as api_client:
        class_names, root_files = _list_tree(api_client, VALIDATION_PREFIX)
        if root_files:
            raise RuntimeError("val.X unexpectedly contains files outside class folders")
        class_names = sorted(Path(name).name for name in class_names)
        if len(class_names) != 100:
            raise RuntimeError(f"expected 100 validation classes, found {len(class_names)}")

        remote_files: list[str] = []
        for class_index, class_name in enumerate(class_names, start=1):
            class_path = f"{VALIDATION_PREFIX}/{class_name}"
            nested, files = _list_tree(api_client, class_path)
            if nested:
                raise RuntimeError(f"unexpected nested directory below {class_path}")
            if len(files) != 50:
                raise RuntimeError(f"expected 50 files below {class_path}, found {len(files)}")
            remote_files.extend(f"{class_path}/{Path(name).name}" for name in files)
            if class_index % 20 == 0:
                print(f"[index] {class_index}/100 classes", flush=True)

        def download_one(remote_path: str) -> str:
            destination = output_dir / Path(remote_path)
            if destination.is_file() and not force:
                return "cached"
            destination.parent.mkdir(parents=True, exist_ok=True)
            request = ApiDownloadDatasetRequest()
            request.owner_slug = DATASET_OWNER
            request.dataset_slug = DATASET_SLUG
            request.dataset_version_number = DATASET_VERSION
            request.file_name = remote_path
            response = api_client.datasets.dataset_api_client.download_dataset(request)
            download_file(
                response,
                str(destination),
                handle,
                extract_auto_compressed_file=True,
            )
            return "downloaded"

        completed = 0
        downloaded = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(download_one, path) for path in remote_files]
            for future in as_completed(futures):
                downloaded += future.result() == "downloaded"
                completed += 1
                if completed % 250 == 0 or completed == len(futures):
                    print(
                        f"[files] {completed}/{len(futures)} | new {downloaded}",
                        flush=True,
                    )


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
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if args.workers <= 0:
        raise ValueError("workers must be positive")

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
        _download_validation_tree(output_dir, workers=args.workers, force=args.force)

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
