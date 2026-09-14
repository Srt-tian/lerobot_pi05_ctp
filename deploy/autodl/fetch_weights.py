"""Fetch the pinned base on AutoDL with four bounded workers and no automatic retries."""

import concurrent.futures
import hashlib
import json
import os
import shutil
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path("/root/autodl-tmp/models/pi05_base")
REVISION = "b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba"
TOTAL = 14467165872
SHA256 = "0eb11ca9587678c1d2ef8cf32807c29f8ce53a2bfdfc1aa4a4c96f16fca59b0f"
URL = f"https://hf-mirror.com/lerobot/pi05_base/resolve/{REVISION}/model.safetensors?download=true"


def verify_and_link(source):
    """Verify the whole file before exposing it to the task-specific init bundle."""
    assert source.stat().st_size == TOTAL
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024**2), b""):
            digest.update(block)
    assert digest.hexdigest() == SHA256, digest.hexdigest()
    final = ROOT / "model.safetensors"
    if source != final:
        source.rename(final)
    target = ROOT.parent / "pi05_base_pens168_init/model.safetensors"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        os.link(final, target)
    assert os.path.samefile(final, target), "The init bundle points to different weights"
    # Pinned, checked-in small metadata files avoid a second network dependency
    # after the large model has already completed and passed its checksum.
    for metadata in (Path(__file__).parent / "base_metadata").glob("*.json"):
        shutil.copyfile(metadata, ROOT / metadata.name)
    report = {"source": "lerobot/pi05_base", "revision": REVISION, "bytes": TOTAL, "sha256": SHA256}
    (ROOT / "source_provenance.json").write_text(json.dumps(report, indent=2))
    print("AUTODL_WEIGHTS_READY", json.dumps(report), flush=True)


def main():
    """Resume only completed byte ranges, then validate the complete model."""
    ROOT.mkdir(parents=True, exist_ok=True)
    final = ROOT / "model.safetensors"
    if final.exists():
        verify_and_link(final)
        return
    partial = ROOT / "model.safetensors.partial"
    manifest_path = ROOT / "download_ranges.json"
    if manifest_path.exists():
        assert partial.exists(), "Range manifest exists without its partial file"
        manifest = json.loads(manifest_path.read_text())
        assert manifest["total_bytes"] == TOTAL
    else:
        partial.touch(exist_ok=True)
        manifest = {"prefix_bytes": partial.stat().st_size, "total_bytes": TOTAL, "completed": []}
        manifest_path.write_text(json.dumps(manifest))
    assert 0 <= manifest["prefix_bytes"] <= TOTAL
    chunk = 128 * 1024**2
    ranges = [
        (start, min(start + chunk, TOTAL) - 1) for start in range(manifest["prefix_bytes"], TOTAL, chunk)
    ]
    assert all(tuple(item) in ranges for item in manifest["completed"])
    assert len({tuple(item) for item in manifest["completed"]}) == len(manifest["completed"])
    pending = [item for item in ranges if list(item) not in manifest["completed"]]
    lock = threading.Lock()
    started = time.monotonic()
    received = 0
    fd = os.open(partial, os.O_RDWR)

    def fetch(item):
        nonlocal received
        start, end = item
        request = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(request, timeout=120) as response:
            assert response.status == 206
            assert response.headers.get("Content-Range") == f"bytes {start}-{end}/{TOTAL}"
            offset = start
            while block := response.read(4 * 1024**2):
                assert offset + len(block) <= end + 1
                written = 0
                while written < len(block):
                    written += os.pwrite(fd, block[written:], offset + written)
                offset += len(block)
                with lock:
                    received += len(block)
            assert offset == end + 1
        os.fsync(fd)
        with lock:
            manifest["completed"].append([start, end])
            temporary = manifest_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(manifest))
            os.replace(temporary, manifest_path)
            done = manifest["prefix_bytes"] + sum(b - a + 1 for a, b in manifest["completed"])
            speed = received / 1024**2 / (time.monotonic() - started)
            print(f"PROGRESS completed={done}/{TOTAL} speed_mib_s={speed:.2f}", flush=True)

    print("START", json.dumps({"workers": 4, "retries": 0, "remaining_chunks": len(pending)}), flush=True)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(fetch, pending))
    finally:
        os.close(fd)
    verify_and_link(partial)


if __name__ == "__main__":
    main()
