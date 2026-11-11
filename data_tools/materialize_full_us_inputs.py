"""Assemble a local Full US dataset from licensed feature and execution caches."""
import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path, PureWindowsPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_tools.verify_full_us_data import (
    MANIFESTS, ROOT, extra_allow, feature_allow, read_json, records, safe_member, verify,
)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def public_metadata(group, name, source):
    if group == "train_validation" and name == "LOCK.json":
        value = read_json(source / name)
        return json_bytes({key: value[key] for key in (
            "schema", "feature_contract", "feature_code", "normalization", "availability",
            "monthly_missingness", "retrospective_roster_quality_screen_through")})
    if group == "train_validation" and name == "monthly/READY.json":
        value = read_json(source / name)
        for record in value["sources"]:
            record["path"] = PureWindowsPath(record["path"]).name
        return json_bytes(value)
    if group == "train_validation" and name == "monthly/MANIFEST.json":
        value = read_json(source / name)
        ready = public_metadata(group, "monthly/READY.json", source)
        for record in value["files"]:
            if record["path"] == "READY.json":
                record.update(bytes=len(ready), sha256=hashlib.sha256(ready).hexdigest())
        return json_bytes(value)
    if group == "test_execution_market" and name == "data/EVIDENCE.json":
        value = read_json(source / name)
        for record in value["sources"].values():
            if "file" in record:
                record["file"] = PureWindowsPath(record["file"]).name
        return json_bytes(value)
    return None


def project(group, source, rows, destination):
    dest = destination / MANIFESTS[group]
    output = []
    for row in sorted(rows, key=lambda r: r["path"]):
        name = row["path"]
        src = safe_member(source, name)
        target = safe_member(dest.parent, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        if src.stat().st_size != row["bytes"]:
            raise ValueError(f"Invalid source size: {src}")
        replacement = public_metadata(group, name, source)
        if replacement is None:
            checksum = hashlib.sha256()
            with src.open("rb") as inp, target.open("xb") as out:
                for block in iter(lambda: inp.read(4 << 20), b""):
                    checksum.update(block)
                    out.write(block)
            original_digest = checksum.hexdigest()
            output.append(dict(path=name, bytes=row["bytes"], sha256=original_digest))
        else:
            original_digest = hashlib.sha256(src.read_bytes()).hexdigest()
            target.write_bytes(replacement)
            output.append(dict(path=name, bytes=len(replacement), sha256=hashlib.sha256(replacement).hexdigest()))
        if original_digest != row["sha256"]:
            raise ValueError(f"Invalid source checksum: {src}")
    payload = json_bytes({"files": output})
    dest.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def inputs(shared, transfer):
    features = transfer / "data/dq-test"
    return {
        "train_validation": (shared, [r for r in records(shared / "MANIFEST.json") if feature_allow(r["path"])]),
        "test_features": (features, [r for r in records(features / "MANIFEST.json") if feature_allow(r["path"])]),
        "test_execution_market": (transfer, [r for r in records(transfer / "SOURCE_MANIFEST.json") if extra_allow(r["path"])]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-shared", required=True, type=Path)
    parser.add_argument("--test-transfer", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "data", help="New dataset directory")
    args = parser.parse_args()
    shared, transfer, out = (path.resolve() for path in (args.training_shared, args.test_transfer, args.out))
    if out.exists() or any(out.is_relative_to(base) for base in (shared, transfer)):
        parser.error("Choose a new destination outside the source directories.")
    groups = inputs(shared, transfer)
    size = sum(r["bytes"] for _, rows in groups.values() for r in rows)
    if shutil.disk_usage(out.parent).free < size + (128 << 20):
        parser.error("Insufficient disk space for the Full-US dataset.")
    with tempfile.TemporaryDirectory(prefix=".full-us-", dir=out.parent) as tmp:
        stage = Path(tmp) / "data"
        for group, (source, rows) in groups.items():
            project(group, source, rows, stage)
            print(f"{group}: verified and exported {len(rows)} files", flush=True)
        verify(stage)
        stage.rename(out)
    print(f"Dataset ready at {out}")


if __name__ == "__main__":
    main()
