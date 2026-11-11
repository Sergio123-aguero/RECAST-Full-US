"""Execution array names and local input-manifest validation."""
import hashlib
import json
from pathlib import Path

ARRAY_KEYS=("asset_ids","target_asset_ids","target_indices","tradable_fill","observed_return",
    "stale_mark","requires_zero_holding","terminal_end","terminal_cash_settlement_observed",
    "receivable_start","receivable_carry","receivable_start_last_mark","receivable_mark_update",
    "mark_returns","receivable_cash_fraction")

DATE_KEYS=("signal_date","execution_date","outcome_date")


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle,"sha256").hexdigest()


def write_new(path,value):
    with Path(path).open("x",encoding="utf-8") as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2,default=str,allow_nan=False)


def safe_path(root,name):
    root=Path(root).resolve()
    candidate=(root/name).resolve()
    if not candidate.is_relative_to(root) or candidate==root:
        raise ValueError("manifest path escapes bundle")
    return candidate


def verify_manifest(root,name="MANIFEST.json"):
    root=Path(root)
    manifest=json.loads((root/name).read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        path=safe_path(root,entry["path"])
        if not path.is_file() or path.stat().st_size!=entry["bytes"] or sha(path)!=entry["sha256"]:
            raise ValueError("portable artifact hash mismatch: "+entry["path"])
    return manifest


def inventory(root):
    root=Path(root)
    return [{"path":p.relative_to(root).as_posix(),"bytes":p.stat().st_size,"sha256":sha(p)}
        for p in sorted(root.rglob("*")) if p.is_file() and p!=root/"MANIFEST.json"]
