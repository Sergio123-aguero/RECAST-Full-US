"""Estimate available RAM from the host and applicable container limits."""
from pathlib import Path
import psutil


def available_memory(*, cgroup_root=Path('/sys/fs/cgroup'), host_available=None):
    host = int(psutil.virtual_memory().available if host_available is None else host_available)
    root = Path(cgroup_root)
    result = dict(host_available_bytes=host, available_bytes=host, cgroup_limit_bytes=None,
        cgroup_current_bytes=None, reclaimable_inactive_file_bytes=0, source='host')
    pairs = ((root/'memory.max', root/'memory.current', root/'memory.stat', 'inactive_file'),
             (root/'memory/memory.limit_in_bytes', root/'memory/memory.usage_in_bytes',
              root/'memory/memory.stat', 'total_inactive_file'))
    for limit_path, current_path, stat_path, key in pairs:
        if not limit_path.is_file():
            continue
        value = limit_path.read_text().strip()
        if value == 'max':
            continue
        limit = int(value)
        # v1 represents no limit by an enormous sentinel.
        if limit >= 1 << 60:
            continue
        current = int(current_path.read_text().strip())
        stats = dict(line.split() for line in stat_path.read_text().splitlines())
        reclaimable = min(current, int(stats.get(key, 0)))
        free = max(0, limit-current+reclaimable)
        if free < result['available_bytes']:
            result.update(available_bytes=free, cgroup_limit_bytes=limit,
                cgroup_current_bytes=current, reclaimable_inactive_file_bytes=reclaimable,
                source=str(limit_path))
    return result
