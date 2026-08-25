"""Reusable utilities shared by model-management and benchmark code."""

import json
import struct
import sys
import tarfile
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from typing import BinaryIO, Callable


def select_model_names(
    arguments: list[str],
    groups: dict[str, tuple[str, ...]],
    selectors: tuple[str, ...],
) -> list[str]:
    """Resolve individual model and family selectors in registry order."""
    if not arguments:
        raise ValueError("Select at least one model, 'yolo', 'small-vlm', or 'all'")
    if "all" in arguments and len(arguments) != 1:
        raise ValueError("Use 'all' alone, or select individual models or families")

    selected = []
    for argument in arguments:
        if argument in groups:
            selected.extend(groups[argument])
        elif argument in selectors:
            selected.append(argument)
        else:
            raise ValueError(f"Unknown model selector: {argument}")
    return list(dict.fromkeys(selected))


def inspect_safetensors(directory: Path) -> dict[str, int]:
    """Validate local safetensors files and count tensors by stored dtype."""
    dtype_counts: Counter[str] = Counter()
    weight_files = sorted(directory.glob("*.safetensors"))
    if not weight_files:
        raise RuntimeError(f"Downloaded snapshot has no safetensors weights: {directory}")

    for weight_file in weight_files:
        with weight_file.open("rb") as stream:
            header_size_bytes = stream.read(8)
            if len(header_size_bytes) != 8:
                raise RuntimeError(f"Invalid safetensors header: {weight_file}")
            header_size = struct.unpack("<Q", header_size_bytes)[0]
            try:
                header = json.loads(stream.read(header_size))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise RuntimeError(f"Invalid safetensors metadata: {weight_file}") from error

        tensors = [value for key, value in header.items() if key != "__metadata__"]
        if not tensors:
            raise RuntimeError(f"Safetensors file contains no tensors: {weight_file}")
        expected_size = 8 + header_size + max(value["data_offsets"][1] for value in tensors)
        if weight_file.stat().st_size != expected_size:
            raise RuntimeError(f"Truncated safetensors file: {weight_file}")
        dtype_counts.update(value["dtype"] for value in tensors)

    return dict(sorted(dtype_counts.items()))


def download_file(
    url: str,
    destination: Path,
    opener: Callable[[str], BinaryIO] | None = None,
) -> None:
    """Stream a URL to a local file without loading it into memory."""
    open_url = opener or urllib.request.urlopen
    with open_url(url) as response, destination.open("wb") as output:
        content_length = response.headers.get("Content-Length")
        total_bytes = int(content_length) if content_length else None
        downloaded_bytes = 0

        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            downloaded_bytes += len(chunk)
            if total_bytes:
                fraction = min(downloaded_bytes / total_bytes, 1)
                completed = round(fraction * 30)
                bar = "#" * completed + "-" * (30 - completed)
                progress = f"\r[{bar}] {fraction:6.1%}"
            else:
                progress = f"\rDownloaded {downloaded_bytes / 1024**2:.1f} MiB"
            print(progress, end="", file=sys.stderr, flush=True)

        print(file=sys.stderr)


def extract_zip_safely(archive: Path, destination: Path) -> None:
    """Extract a ZIP archive after rejecting paths outside the destination."""
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as zip_archive:
        for member in zip_archive.infolist():
            member_path = (destination / member.filename).resolve()
            if not member_path.is_relative_to(destination):
                raise RuntimeError(f"Unsafe ZIP member path: {member.filename}")
        zip_archive.extractall(destination)


def extract_tar_prefix_safely(archive: Path, destination: Path, prefix: str) -> None:
    """Extract one TAR directory tree after rejecting unsafe member paths."""
    destination = destination.resolve()
    with tarfile.open(archive) as tar_archive:
        members = []
        for member in tar_archive.getmembers():
            if member.name != prefix and not member.name.startswith(f"{prefix}/"):
                continue
            member_path = (destination / member.name).resolve()
            if not member_path.is_relative_to(destination):
                raise RuntimeError(f"Unsafe TAR member path: {member.name}")
            if member.issym() or member.islnk():
                raise RuntimeError(f"Archive links are not allowed: {member.name}")
            members.append(member)
        tar_archive.extractall(destination, members=members)
