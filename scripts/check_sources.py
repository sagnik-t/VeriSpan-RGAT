"""
scripts/check_sources.py

Checks every known data source for FEVER, SciFact, and WiCE.
Run this once to find out what's accessible from your machine.

Usage:
    python scripts/check_sources.py
"""

import sys
import urllib.request
import urllib.error

SOURCES = {
    "FEVER": [
        ("fever.ai S3 train",       "https://s3-eu-west-1.amazonaws.com/fever.public/train.jsonl"),
        ("fever.ai S3 dev",         "https://s3-eu-west-1.amazonaws.com/fever.public/paper_dev.jsonl"),
        ("fever.ai S3 wiki",        "https://s3-eu-west-1.amazonaws.com/fever.public/wiki-pages.zip"),
        ("fever.ai direct train",   "https://fever.ai/download/fever/train.jsonl"),
        ("fever.ai direct dev",     "https://fever.ai/download/fever/shared_task_dev.jsonl"),
        ("HF FEVER parquet train",  "https://huggingface.co/datasets/fever/resolve/main/data/train-00000-of-00002.parquet"),
        ("HF FEVER parquet dev",    "https://huggingface.co/datasets/fever/resolve/main/data/validation-00000-of-00001.parquet"),
        ("Zenodo FEVER",            "https://zenodo.org/record/1473641/files/train.jsonl"),
    ],
    "SciFact": [
        ("AI2 S3 scifact",          "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"),
        ("AI2 S3 alt",              "https://ai2-public-datasets.s3-us-west-2.amazonaws.com/scifact/scifact.tar.gz"),
        ("GitHub release data",     "https://github.com/allenai/scifact/releases/download/v1.0/data.tar.gz"),
        ("GitHub raw claims_dev",   "https://raw.githubusercontent.com/allenai/scifact/main/data/claims_dev.jsonl"),
        ("GitHub raw corpus",       "https://raw.githubusercontent.com/allenai/scifact/main/data/corpus.jsonl"),
        ("HF SciFact parquet test", "https://huggingface.co/datasets/allenai/scifact/resolve/main/data/test-00000-of-00001.parquet"),
        ("HF SciFact parquet train","https://huggingface.co/datasets/allenai/scifact/resolve/main/data/train-00000-of-00001.parquet"),
    ],
    "WiCE": [
        ("ryokamoi/wice HF",        "https://huggingface.co/datasets/ryokamoi/wice/resolve/main/README.md"),
        ("Babelscape/wice HF",      "https://huggingface.co/datasets/Babelscape/wice/resolve/main/README.md"),
        ("sihaochen/wice HF",       "https://huggingface.co/datasets/sihaochen/wice/resolve/main/README.md"),
        ("rcadg/wice HF",           "https://huggingface.co/datasets/rcadg/wice/resolve/main/README.md"),
        ("GitHub ryokamoi wice",    "https://raw.githubusercontent.com/ryokamoi/wice/main/README.md"),
        ("GitHub ryokamoi wice2",   "https://raw.githubusercontent.com/ryokamoi/wice/master/README.md"),
        ("HF wice-nli",             "https://huggingface.co/datasets/wice-nli/wice/resolve/main/README.md"),
    ],
}


def check(url: str, timeout: int = 8) -> tuple[int, str]:
    try:
        req = urllib.request.Request(
            url,
            method="HEAD",
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, ""
    except urllib.error.HTTPError as e:
        return e.code, ""
    except Exception as e:
        return 0, str(type(e).__name__)


def main():
    print("Checking data source accessibility...\n")
    working = {}

    for dataset, sources in SOURCES.items():
        print(f"── {dataset} " + "─" * (50 - len(dataset)))
        working[dataset] = []
        for name, url in sources:
            code, err = check(url)
            status = "✓ OK " if code == 200 else f"✗ {code}" if code else f"✗ ERR"
            print(f"  {status}  {name}")
            if code == 200:
                working[dataset].append((name, url))
        print()

    print("=" * 60)
    print("SUMMARY — working sources:")
    any_working = False
    for dataset, sources in working.items():
        if sources:
            any_working = True
            print(f"\n  {dataset}:")
            for name, url in sources:
                print(f"    ✓ {name}")
                print(f"      {url}")

    if not any_working:
        print("\n  No working sources found.")
        print("  Options:")
        print("  1. Academic Torrents (requires torrent client):")
        print("     FEVER: https://academictorrents.com/details/fcfcbe3dc6b3a36e6dd19bed26fbe089c8aabb59")
        print("  2. Kaggle (requires free account + kaggle CLI):")
        print("     kaggle datasets download harunshimanto/fever-fact-verification-dataset")
        print("  3. Set HF_TOKEN and retry:")
        print("     export HF_TOKEN=your_token_here")
        print("     python scripts/check_sources.py")

    print()


if __name__ == "__main__":
    main()
