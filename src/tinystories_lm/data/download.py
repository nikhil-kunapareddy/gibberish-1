"""Fetch the TinyStories corpora from the HuggingFace CDN and cache them.

Uses stdlib `urllib` rather than `datasets`/`huggingface_hub`: the corpora are
plain text files behind a CDN that honours HTTP range requests, which is also
how `max_download_bytes` keeps `make smoke` from pulling 1.9GB onto a laptop.

Datasets: roneneldan/TinyStories and roneneldan/TinyStories-Instruct,
licensed CDLA-Sharing-1.0 (free to use and redistribute with attribution).

Caching: a file is re-downloaded only if it is missing, or if a previous run
fetched fewer bytes than this run asks for. The sidecar `<name>.download.json`
records what was fetched.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

HF_URL = "https://huggingface.co/datasets/{repo}/resolve/main/{filename}"
CHUNK = 1 << 20


@dataclass(frozen=True)
class DownloadedFile:
    path: Path
    bytes_read: int
    total_bytes: int | None
    truncated: bool  # True when we deliberately fetched a prefix of the file

    @property
    def complete(self) -> bool:
        return not self.truncated


def _sidecar(dest: Path) -> Path:
    return dest.with_suffix(dest.suffix + ".download.json")


def _read_sidecar(dest: Path) -> dict | None:
    p = _sidecar(dest)
    if not (p.exists() and dest.exists()):
        return None
    try:
        meta = json.loads(p.read_text())
    except json.JSONDecodeError:
        return None
    if meta.get("bytes_read") != dest.stat().st_size:
        return None  # interrupted download; refetch
    return meta


def download_file(
    repo: str,
    filename: str,
    dest_dir: str | Path,
    max_bytes: int | None = None,
    force: bool = False,
    quiet: bool = False,
) -> DownloadedFile:
    """Download (a prefix of) one file from a HF dataset repo, with caching."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / Path(filename).name

    cached = None if force else _read_sidecar(dest)
    if cached is not None:
        have, total = cached["bytes_read"], cached.get("total_bytes")
        enough = (max_bytes is not None and have >= max_bytes) or (
            total is not None and have >= total
        )
        if enough:
            if not quiet:
                print(f"  cached: {dest} ({have / 1e6:.1f} MB)")
            return DownloadedFile(dest, have, total, truncated=bool(total and have < total))

    url = HF_URL.format(repo=repo, filename=urllib.parse.quote(filename))
    headers = {"User-Agent": "tinystories-lm/0.1"}
    if max_bytes is not None:
        headers["Range"] = f"bytes=0-{max_bytes - 1}"

    if not quiet:
        limit = f" (first {max_bytes / 1e6:.0f} MB)" if max_bytes else ""
        print(f"  downloading {repo}/{filename}{limit} -> {dest}")

    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        content_range = response.headers.get("Content-Range")
        total = int(content_range.split("/")[-1]) if content_range else _int_or_none(
            response.headers.get("Content-Length")
        )
        expect = min(max_bytes, total) if (max_bytes and total) else (max_bytes or total)
        read = 0
        tmp = dest.with_suffix(dest.suffix + ".part")
        with tmp.open("wb") as out, tqdm(
            total=expect, unit="B", unit_scale=True, disable=quiet, leave=False
        ) as bar:
            while chunk := response.read(CHUNK):
                out.write(chunk)
                read += len(chunk)
                bar.update(len(chunk))
        tmp.replace(dest)

    truncated = bool(total and read < total)
    _sidecar(dest).write_text(
        json.dumps(
            {"repo": repo, "filename": filename, "url": url, "bytes_read": read,
             "total_bytes": total, "truncated": truncated},
            indent=2,
        )
    )
    return DownloadedFile(dest, read, total, truncated)


def _int_or_none(v: str | None) -> int | None:
    return int(v) if v is not None else None


def ensure_corpora(cfg, quiet: bool = False) -> dict[str, DownloadedFile]:
    """Fetch both corpora named by a DataConfig. Returns {"stories", "instruct"}."""
    raw_dir = Path(cfg.data_dir) / "raw"
    return {
        "stories": download_file(
            cfg.dataset, cfg.source_file, raw_dir, cfg.max_download_bytes, quiet=quiet
        ),
        "instruct": download_file(
            cfg.instruct_dataset,
            cfg.instruct_source_file,
            raw_dir,
            cfg.max_download_bytes,
            quiet=quiet,
        ),
    }
