"""Download the SonoGym expert lerobot dataset from Hugging Face.

Usage:
    python scripts/download_lerobot_dataset.py                # default configs' datasets (guidance + surgery, ~8.6GB)
    python scripts/download_lerobot_dataset.py --all           # all 8 sub-datasets (~34GB)
    python scripts/download_lerobot_dataset.py --datasets Isaac-robot-US-guidance-v0-single-net

Requires: pip install huggingface_hub
"""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "yunkao/SonoGym_lerobot_dataset"

ALL_DATASETS = [
    "Isaac-robot-US-guidance-v0-single",
    "Isaac-robot-US-guidance-v0-single-net",
    "Isaac-robot-US-guidance-5-models-v0",
    "Isaac-robot-US-guided-surgery-v0-single-new",
    "Isaac-robot-US-guided-surgery-v0-single-net-new",
    "Isaac-robot-US-guided-surgery-v0-5-net",
    "Isaac-robot-US-guided-surgery-v0-5",
    "Isaac-robot-US-guided-surgery-5-models-v0",
]

# Datasets referenced by configs/training/train_{guidance,surgery}_{act,diffusion}_cfg.json
DEFAULT_DATASETS = [
    "Isaac-robot-US-guidance-v0-single",
    "Isaac-robot-US-guided-surgery-v0-single-new",
]

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOCAL_DIR = REPO_ROOT / "data" / "raw" / "lerobot-dataset"

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument(
    "--datasets",
    nargs="+",
    choices=ALL_DATASETS,
    default=None,
    help="Specific sub-dataset(s) to download. Defaults to the datasets used by the shipped lerobot train configs.",
)
parser.add_argument("--all", action="store_true", help="Download all 8 sub-datasets (~34GB).")
parser.add_argument(
    "--local-dir",
    type=Path,
    default=DEFAULT_LOCAL_DIR,
    help=f"Target directory (default: {DEFAULT_LOCAL_DIR}).",
)
parser.add_argument("--max-workers", type=int, default=8)
args = parser.parse_args()

if args.all:
    datasets = ALL_DATASETS
else:
    datasets = args.datasets or DEFAULT_DATASETS

allow_patterns = [f"{name}/*" for name in datasets]

print(f"Downloading from {REPO_ID}:")
for name in datasets:
    print(f"  - {name}")
print(f"Target dir: {args.local_dir}")

snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    local_dir=str(args.local_dir),
    allow_patterns=allow_patterns,
    max_workers=args.max_workers,
)

print("Download complete.")
