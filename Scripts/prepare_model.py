#!/usr/bin/env python3
"""Fetch pinned model artifacts once, verify SHA-256, and atomically install them for bundling."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "Scripts" / "model-manifest.json"
MODEL = ROOT / "Models" / "SmolVLM2"


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def matches(path, artifact):
    return (path.is_file() and path.stat().st_size == artifact["size"]
            and digest(path) == artifact["sha256"])


@contextlib.contextmanager
def exclusive_lock(path):
    # Serialize simultaneous Xcode builds without leaving a stale lock after a crash.
    with path.open("a+b") as lock:
        if os.name == "nt":
            import msvcrt
            lock.write(b"0")
            lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock, fcntl.LOCK_UN)


def prepare(verify_only=False):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    MODEL.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(MODEL.parent / ".prepare-model.lock"):
        for artifact in manifest["files"]:
            name = artifact["name"]
            if Path(name).name != name or name in (".", ".."):
                raise ValueError("Invalid artifact filename")
            target = MODEL / name
            if matches(target, artifact):
                continue
            if verify_only:
                raise ValueError(f"Missing or corrupt model artifact: {name}")
            print(f"Fetching {name} ({artifact['size'] / 1e6:.1f} MB)…", flush=True)
            url = f"https://huggingface.co/{manifest['repository']}/resolve/{manifest['revision']}/{name}"
            handle, temporary = tempfile.mkstemp(prefix=".download-", dir=MODEL.parent)
            os.close(handle)
            temporary = Path(temporary)
            try:
                subprocess.run([
                    "curl", "--fail", "--location", "--silent", "--show-error",
                    "--retry", "3", "--connect-timeout", "30", "--max-time", "1800",
                    "--output", str(temporary), url,
                ], check=True)
                if not matches(temporary, artifact):
                    raise ValueError(f"Checksum mismatch for {name}; refusing to bundle it")
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        # Only mark complete after all files pass. No partial download can look installed.
        marker = MODEL / "verified.json"
        marker.write_text(json.dumps({"revision": manifest["revision"]}), encoding="utf-8")
    print("SmolVLM2 model verified and ready to bundle.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    try:
        prepare(args.verify_only)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"error: Model preparation failed: {error}", file=sys.stderr)
        sys.exit(1)
