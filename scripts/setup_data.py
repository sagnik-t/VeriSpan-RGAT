"""
scripts/setup_data.py — One-shot data setup for VeriSpan-RGAT.

Confirmed working sources (verified on target machine):
    FEVER   : https://fever.ai/download/fever/
    SciFact : https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz
    WiCE    : https://github.com/ryokamoi/wice

Usage
-----
    python scripts/setup_data.py                   # download everything
    python scripts/setup_data.py --skip_wiki       # skip FEVER wiki pages (~1.8 GB)
    python scripts/setup_data.py --verify_only     # check existing files only
    python scripts/setup_data.py --only fever      # one dataset at a time
    python scripts/setup_data.py --only scifact
    python scripts/setup_data.py --only wice
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("setup_data")

_HEADERS = {"User-Agent": "Mozilla/5.0 VeriSpan-RGAT/1.0"}


# ── Download primitives ───────────────────────────────────────────────────────

def _head(url: str, timeout: int = 8) -> int:
    try:
        req = urllib.request.Request(url, method="HEAD", headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return 0


def _fetch_text(url: str, timeout: int = 10) -> str | None:
    try:
        req = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", errors="replace")
    except Exception:
        return None


def _download(url: str, dest: Path, label: str = "") -> bool:
    """Stream-download url → dest with MB progress. Returns True on success."""
    if dest.exists():
        logger.info(f"  Already exists: {dest.name}")
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    label = label or dest.name
    logger.info(f"  Downloading {label} ...")

    try:
        req = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=120) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            with open(dest, "wb") as f:
                while chunk := resp.read(1024 * 512):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded * 100 // total
                        print(f"\r    {pct:3d}%  {downloaded/1e6:.1f} MB", end="", flush=True)
            print()
        logger.info(f"  ✓ {dest.name}")
        return True
    except Exception as e:
        logger.error(f"  ✗ {label}: {e}")
        if dest.exists():
            dest.unlink()
        return False


# ── FEVER ─────────────────────────────────────────────────────────────────────

_FEVER_BASE = "https://fever.ai/download/fever"

_FEVER_FILES = {
    "train.jsonl":           f"{_FEVER_BASE}/train.jsonl",
    "paper_dev.jsonl":       f"{_FEVER_BASE}/paper_dev.jsonl",
    "paper_test.jsonl":      f"{_FEVER_BASE}/paper_test.jsonl",
    "shared_task_dev.jsonl": f"{_FEVER_BASE}/shared_task_dev.jsonl",
}
_FEVER_WIKI_URL = f"{_FEVER_BASE}/wiki-pages.zip"


def download_fever(fever_dir: Path, skip_wiki: bool = False) -> bool:
    fever_dir.mkdir(parents=True, exist_ok=True)
    ok = True

    logger.info("Downloading FEVER claim files ...")
    for fname, url in _FEVER_FILES.items():
        if not _download(url, fever_dir / fname, fname):
            ok = False

    if skip_wiki:
        logger.info("Skipping wiki pages (--skip_wiki).")
        return ok

    wiki_dir = fever_dir / "wiki-pages"
    if wiki_dir.exists() and any(wiki_dir.glob("wiki-*.jsonl")):
        logger.info("  wiki-pages/ already present — skipping.")
        return ok

    logger.info("\nDownloading FEVER wiki pages (~1.8 GB) ...")
    zip_path = fever_dir / "wiki-pages.zip"
    if _download(_FEVER_WIKI_URL, zip_path, "wiki-pages.zip"):
        logger.info("  Extracting wiki-pages.zip ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(fever_dir)
        zip_path.unlink()
        logger.info("  ✓ wiki-pages/ extracted")
    else:
        logger.warning(
            "  wiki-pages.zip download failed.\n"
            "  Training will only use NEI examples until wiki pages are available.\n"
            "  Retry with: python scripts/setup_data.py --only fever"
        )
        ok = False

    return ok


def verify_fever(fever_dir: Path) -> bool:
    ok = True
    for fname in _FEVER_FILES:
        p = fever_dir / fname
        if p.exists():
            logger.info(f"  ✓ {fname}  ({p.stat().st_size / 1e6:.1f} MB)")
        else:
            logger.error(f"  ✗ {fname} — MISSING")
            ok = False

    wiki_dir = fever_dir / "wiki-pages"
    n = len(list(wiki_dir.glob("wiki-*.jsonl"))) if wiki_dir.exists() else 0
    if n > 0:
        logger.info(f"  ✓ wiki-pages/ ({n} files)")
    else:
        logger.warning("  ✗ wiki-pages/ — MISSING (needed for evidence resolution)")
    return ok


# ── SciFact ───────────────────────────────────────────────────────────────────

_SCIFACT_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"

# tar member path → local filename
_SCIFACT_MAP = {
    "data/claims_train.jsonl": "claims_train.jsonl",
    "data/claims_dev.jsonl":   "claims_dev.jsonl",
    "data/corpus.jsonl":       "corpus.jsonl",
    "data/claims_test.jsonl":  "claims_test.jsonl",
}


def download_scifact(scifact_dir: Path) -> bool:
    scifact_dir.mkdir(parents=True, exist_ok=True)

    expected = ["claims_train.jsonl", "claims_dev.jsonl", "corpus.jsonl"]
    if all((scifact_dir / f).exists() for f in expected):
        logger.info("  SciFact already extracted — skipping.")
        return True

    tar_path = scifact_dir / "scifact.tar.gz"
    if not _download(_SCIFACT_URL, tar_path, "scifact.tar.gz"):
        return False

    logger.info("  Extracting scifact.tar.gz ...")
    extracted = []
    with tarfile.open(tar_path, "r:gz") as tf:
        members = tf.getmembers()
        logger.info(f"  Archive contains {len(members)} files:")
        for m in members:
            logger.info(f"    {m.name}")
            dest_name = _SCIFACT_MAP.get(m.name)
            if dest_name:
                # Extract to a temp name then rename
                m_copy = tarfile.TarInfo(name=dest_name)
                m_copy.size = m.size
                f = tf.extractfile(m)
                if f:
                    with open(scifact_dir / dest_name, "wb") as out:
                        out.write(f.read())
                    extracted.append(dest_name)
                    logger.info(f"    → {dest_name}")

    tar_path.unlink()
    logger.info(f"  ✓ SciFact ready ({len(extracted)} files extracted)")
    return len(extracted) > 0


def verify_scifact(scifact_dir: Path) -> bool:
    ok = True
    for fname in ["claims_train.jsonl", "claims_dev.jsonl", "corpus.jsonl"]:
        p = scifact_dir / fname
        if p.exists():
            logger.info(f"  ✓ {fname}  ({p.stat().st_size / 1e6:.1f} MB)")
        else:
            logger.error(f"  ✗ {fname} — MISSING")
            ok = False
    return ok


# ── WiCE ─────────────────────────────────────────────────────────────────────

_WICE_RAW_BASE = "https://raw.githubusercontent.com/ryokamoi/wice"
_WICE_BRANCHES = ["main", "master"]

# Candidate paths to probe in the repo
_WICE_PROBE_PATHS = [
    "data/train.jsonl",       "data/dev.jsonl",       "data/test.jsonl",
    "data/wice_train.jsonl",  "data/wice_dev.jsonl",  "data/wice_test.jsonl",
    "dataset/train.jsonl",    "dataset/dev.jsonl",    "dataset/test.jsonl",
    "wice/train.jsonl",       "wice/dev.jsonl",       "wice/test.jsonl",
    "train.jsonl",            "dev.jsonl",            "test.jsonl",
    "wice_train.jsonl",       "wice_dev.jsonl",       "wice_test.jsonl",
]


def download_wice(wice_dir: Path) -> bool:
    wice_dir.mkdir(parents=True, exist_ok=True)

    # Check if already present
    if list(wice_dir.glob("*.jsonl")):
        logger.info("  WiCE files already present — skipping.")
        return True

    # The ryokamoi/wice repo stores data via git — clone it directly
    repo_url = "https://github.com/ryokamoi/wice.git"
    clone_dir = wice_dir / "_clone"

    logger.info(f"  Cloning ryokamoi/wice ...")
    import subprocess
    result = subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(clone_dir)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        logger.error(f"  git clone failed:\n{result.stderr}")
        return False

    # Find and copy all JSONL files from the clone
    copied = []
    for jsonl in clone_dir.rglob("*.jsonl"):
        # Skip cross-validation folds
        if "cross_validation" in str(jsonl) or "fold" in str(jsonl):
            continue
        dest = wice_dir / jsonl.name
        if not dest.exists():
            import shutil
            shutil.copy2(jsonl, dest)
            copied.append(jsonl.name)
            logger.info(f"  Copied: {jsonl.name}")

    # Clean up clone
    import shutil
    shutil.rmtree(clone_dir)

    if copied:
        logger.info(f"  ✓ WiCE ready ({len(copied)} files)")
        return True

    logger.error(
        "\n  No JSONL files found in ryokamoi/wice after cloning.\n"
        "  The dataset may require manual download.\n"
        "  Please visit https://github.com/ryokamoi/wice and follow instructions.\n"
        f"  Place data files in: {wice_dir.resolve()}"
    )
    return False


def verify_wice(wice_dir: Path) -> bool:
    files = list(wice_dir.glob("*.jsonl")) + list(wice_dir.glob("*.json"))
    if files:
        for f in sorted(files):
            logger.info(f"  ✓ {f.name}  ({f.stat().st_size / 1e6:.1f} MB)")
        return True
    logger.error(f"  ✗ No data files in {wice_dir}")
    return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Set up VeriSpan-RGAT datasets")
    p.add_argument("--data_dir",    default="data")
    p.add_argument("--skip_wiki",   action="store_true",
                   help="Skip FEVER wiki pages (~1.8 GB unzipped).")
    p.add_argument("--verify_only", action="store_true")
    p.add_argument("--only",        choices=["fever", "scifact", "wice"])
    args = p.parse_args()

    base        = Path(args.data_dir)
    fever_dir   = base / "raw" / "fever"
    scifact_dir = base / "raw" / "scifact"
    wice_dir    = base / "raw" / "wice"
    (base / "processed").mkdir(parents=True, exist_ok=True)

    results: dict[str, bool] = {}

    if not args.only or args.only == "fever":
        logger.info("\n── FEVER ──────────────────────────────────────────────")
        if not args.verify_only:
            download_fever(fever_dir, skip_wiki=args.skip_wiki)
        results["FEVER"] = verify_fever(fever_dir)

    if not args.only or args.only == "scifact":
        logger.info("\n── SciFact ─────────────────────────────────────────────")
        if not args.verify_only:
            download_scifact(scifact_dir)
        results["SciFact"] = verify_scifact(scifact_dir)

    if not args.only or args.only == "wice":
        logger.info("\n── WiCE ────────────────────────────────────────────────")
        if not args.verify_only:
            download_wice(wice_dir)
        results["WiCE"] = verify_wice(wice_dir)

    logger.info("\n" + "=" * 60)
    for name, ok in results.items():
        logger.info(f"  {'✓' if ok else '✗'} {name}")
    logger.info("=" * 60)

    if all(results.values()):
        logger.info("All datasets ready.")
    else:
        logger.warning("Some datasets incomplete — see errors above.")


if __name__ == "__main__":
    main()
