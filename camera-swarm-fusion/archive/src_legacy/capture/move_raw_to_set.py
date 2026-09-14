#!/usr/bin/env python3
import shutil, sys
from pathlib import Path

# Usage: python3 src/move_raw_to_set.py [set_name]
set_name = sys.argv[1] if len(sys.argv) > 1 else "checkerboard_test"

root = Path(__file__).resolve().parents[2]
src_dir = root / "data" / "raw"
dst_dir = src_dir / set_name
dst_dir.mkdir(parents=True, exist_ok=True)

# file types to move
exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

moved = 0
skipped = 0

def unique_path(dst: Path) -> Path:
    if not dst.exists():
        return dst
    stem, suffix = dst.stem, dst.suffix
    i = 1
    while True:
        cand = dst.with_name(f"{stem}_{i}{suffix}")
        if not cand.exists():
            return cand
        i += 1

for p in sorted(src_dir.iterdir()):
    if p.is_file() and p.suffix.lower() in exts:
        # Only move files that are directly in data/raw (not already in a subfolder)
        if p.parent != src_dir:
            continue
        target = unique_path(dst_dir / p.name)
        shutil.move(str(p), str(target))
        print(f"Moved: {p.name} -> {target.relative_to(root)}")
        moved += 1
    else:
        skipped += 1

print(f"\nDone. Moved {moved} file(s) into {dst_dir}.")
