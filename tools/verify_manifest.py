"""Verify release file bytes against the committed SHA-256 inventory."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    count = 0
    for line in (ROOT / "MANIFEST.sha256").read_text().splitlines():
        digest, filename = line.split("  ", 1)
        path = ROOT / filename
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise SystemExit(f"Checksum mismatch: {filename}")
        count += 1
    print(f"Verified {count} file checksums")
