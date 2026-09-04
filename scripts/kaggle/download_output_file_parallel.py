from __future__ import annotations

import argparse
import concurrent.futures
import math
import time
from pathlib import Path

import requests
from kagglehub.clients import build_kaggle_client
from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest


def signed_output_url(owner: str, notebook: str, version: int, remote_path: str):
    request = ApiDownloadKernelOutputRequest()
    request.owner_slug = owner
    request.kernel_slug = notebook
    request.version_number = version
    request.file_path = remote_path
    return build_kaggle_client().kernels.kernels_api_client.download_kernel_output(request)


def download_part(
    url: str, start: int, stop: int, path: Path, *, retries: int = 5
) -> None:
    expected = stop - start + 1
    if path.exists() and path.stat().st_size == expected:
        return
    temporary = path.with_suffix(path.suffix + ".partial")
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with requests.get(
                url,
                headers={"Range": f"bytes={start}-{stop}"},
                stream=True,
                timeout=(30, 120),
            ) as response:
                response.raise_for_status()
                if response.status_code != 206:
                    raise RuntimeError(
                        f"server ignored range request: {response.status_code}"
                    )
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            received = temporary.stat().st_size
            if received != expected:
                raise RuntimeError(
                    f"short range {start}-{stop}: {received} != {expected}"
                )
            temporary.replace(path)
            return
        except (requests.RequestException, RuntimeError) as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(
        f"range {start}-{stop} failed after {retries} attempts"
    ) from last_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner", required=True)
    parser.add_argument("--notebook", required=True)
    parser.add_argument("--version", required=True, type=int)
    parser.add_argument("--remote-path", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument(
        "--part-size-kib",
        type=int,
        default=None,
        help="Optional fixed range size; useful for unstable large-file downloads.",
    )
    args = parser.parse_args()
    if args.workers <= 0 or (
        args.part_size_kib is not None and args.part_size_kib <= 0
    ):
        parser.error("workers and part-size-kib must be positive")

    response = signed_output_url(
        args.owner, args.notebook, args.version, args.remote_path
    )
    total = int(response.headers["Content-Length"])
    response.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    parts_dir = args.output.parent / f".{args.output.name}.parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    part_size = (
        int(args.part_size_kib) * 1024
        if args.part_size_kib is not None
        else math.ceil(total / args.workers)
    )
    ranges = []
    part_count = math.ceil(total / part_size)
    for index in range(part_count):
        start = index * part_size
        if start >= total:
            break
        stop = min(total - 1, (index + 1) * part_size - 1)
        ranges.append((index, start, stop, parts_dir / f"part-{index:03d}"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                download_part,
                response.url,
                start,
                stop,
                path,
                retries=args.retries,
            ): index
            for index, start, stop, path in ranges
        }
        for future in concurrent.futures.as_completed(futures):
            index = futures[future]
            future.result()
            print(f"completed range {index + 1}/{len(ranges)}", flush=True)

    assembled = args.output.with_suffix(args.output.suffix + ".assembled")
    with assembled.open("wb") as destination:
        for _, _, _, part in ranges:
            with part.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    destination.write(chunk)
    if assembled.stat().st_size != total:
        raise RuntimeError(f"assembled size {assembled.stat().st_size} != {total}")
    assembled.replace(args.output)
    print(f"saved {args.output} ({total} bytes)")


if __name__ == "__main__":
    main()
