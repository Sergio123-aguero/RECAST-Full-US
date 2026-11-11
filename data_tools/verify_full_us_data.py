"""Verify the Full US panels, execution inputs and market data against local manifests."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = {
    "train_validation": "train_validation/shared/MANIFEST.json",
    "test_features": "test/data/dq-test/MANIFEST.json",
    "test_execution_market": "test/EXECUTION_MANIFEST.json",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def feature_allow(name):
    return (name in {"calendar.json", "READY.json", "LOCK.json", "rosters/US_FIXED.npy", "rosters/US_FIXED.parquet"}
            or name.startswith(("features/", "monthly/", "execution/US_FIXED/")))


def extra_allow(name):
    return (name in {"data/EVIDENCE.json", "data/common-carry/COMMON_CARRY_BRIDGE.json", "market/market.npz", "market/MARKET.json"}
            or name.startswith("data/execution/US_FIXED/"))


def safe_member(folder, name):
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or ".." in path.parts or "\\" in name
            or ":" in name or path.as_posix() != name):
        raise ValueError(f"Invalid relative file name: {name}")
    candidate = folder / path
    if not candidate.resolve().is_relative_to(folder.resolve()):
        raise ValueError(f"File outside the dataset: {name}")
    return candidate


def records(path):
    rows = read_json(path)["files"]
    if not rows or len({r["path"] for r in rows}) != len(rows):
        raise ValueError(f"Empty or duplicate file list in {path}")
    return rows


def verify(root, split="all"):
    if split not in ("train_validation", "test", "all"):
        raise ValueError(f"Unknown data split: {split}")
    report = {}
    root = Path(root)
    expected = set()
    for group, name in MANIFESTS.items():
        if split != "all" and (group == "train_validation") != (split == "train_validation"):
            continue
        manifest = root / name
        expected.add(manifest)
        seal = digest(manifest)
        rows = records(manifest)
        if group == "test_execution_market":
            required = {"data/EVIDENCE.json", "data/common-carry/COMMON_CARRY_BRIDGE.json",
                        "market/market.npz", "market/MARKET.json"}
            allowed = extra_allow
        else:
            required = {"READY.json", "calendar.json", "rosters/US_FIXED.npy"}
            if group == "train_validation":
                required.add("LOCK.json")
            allowed = feature_allow
        if not required.issubset({r["path"] for r in rows}):
            raise ValueError(f"Missing required inputs in {manifest}")
        for row in rows:
            if not allowed(row["path"]):
                raise ValueError(f"Unexpected {group} member: {row['path']}")
            file = safe_member(manifest.parent, row["path"])
            expected.add(file)
            if not file.is_file() or file.stat().st_size != row["bytes"] or digest(file) != row["sha256"]:
                raise ValueError(f"Missing or changed data file: {file}")
        if digest(manifest) != seal:
            raise ValueError(f"Manifest changed during validation: {manifest}")
        report[group] = dict(verified_files=len(rows), verified_bytes=sum(r["bytes"] for r in rows),
                             manifest_sha256=seal)
    folders = (root,) if split == "all" else (root / split,)
    present = {file for folder in folders for file in folder.rglob("*") if file.is_file()}
    if present != expected:
        extras = sorted(str(p.relative_to(root)) for p in present - expected)
        raise ValueError(f"Unexpected files in the Full-US dataset: {extras[:5]}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data")
    parser.add_argument("--split", choices=("train_validation", "test", "all"), default="all")
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.split), indent=2))
