"""Download Imagenette (10 easy ImageNet classes, ~325 MB, free, no login).

Run:  python scripts/01_get_data.py
If the download fails (proxy, firewall), download the file manually in a browser from
    https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
and run:  python scripts/01_get_data.py --archive path\\to\\imagenette2-320.tgz
"""
import argparse
import sys
import tarfile
import urllib.request
from pathlib import Path

from common import IMAGENETTE, Paths

URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"


def safe_extract(tar: tarfile.TarFile, dest: Path):
    dest = dest.resolve()
    for m in tar.getmembers():
        target = (dest / m.name).resolve()
        if dest not in target.parents and target != dest:
            raise RuntimeError(f"Refusing to extract outside {dest}: {m.name}")
    tar.extractall(dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--archive", default=None, help="Use an already-downloaded .tgz instead of downloading")
    a = ap.parse_args()
    paths = Paths(a.work_dir).make()

    if (paths.imagenette / "val").is_dir():
        print(f"Already present: {paths.imagenette}")
    else:
        archive = Path(a.archive) if a.archive else paths.data / "imagenette2-320.tgz"
        if not archive.exists():
            print(f"Downloading {URL}")

            def hook(blocks, block_size, total):
                if total > 0:
                    done = min(blocks * block_size, total)
                    print(f"\r  {done / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)

            urllib.request.urlretrieve(URL, archive, hook)
            print()
        print("Extracting ...")
        with tarfile.open(archive) as tar:
            safe_extract(tar, paths.data)

    missing = [w for w in IMAGENETTE if not (paths.imagenette / "val" / w).is_dir()]
    if missing:
        sys.exit(f"Unexpected layout, missing class folders: {missing}")
    n_train = sum(1 for _ in (paths.imagenette / "train").rglob("*.JPEG"))
    n_val = sum(1 for _ in (paths.imagenette / "val").rglob("*.JPEG"))
    print(f"OK: {n_train} train images, {n_val} validation images in {paths.imagenette}")


if __name__ == "__main__":
    main()
