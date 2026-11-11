"""Fresh, bounded tensor snapshots for a SINGLE content-hash invocation.

No hashes, host snapshots, tensor identities or graphs survive the call. In
particular, version counters are NOT used to skip reads on replay: mutations
through numpy or Tensor.data must still invalidate the input signature. CUDA
tensors of one dtype/device share a transfer, not an arithmetic computation.
"""
import torch


@torch.no_grad()
def tensor_snapshots(value, *, max_batch_bytes=8*1024*1024, include_cpu=False, statistics=None):
    if type(max_batch_bytes) is not int or max_batch_bytes < 1:
        raise ValueError('positive integer transfer bound required')
    tensors, occurrences = {}, 0

    def collect(x):
        nonlocal occurrences
        if isinstance(x, torch.Tensor):
            if x.device.type == 'cuda' or include_cpu:
                occurrences += 1
                tensors.setdefault(id(x), x)
        elif isinstance(x, dict):
            for key in sorted(x):
                collect(x[key])
        elif isinstance(x, (tuple, list)):
            for item in x:
                collect(item)

    collect(value)
    arrays, buckets, sizes = {}, {}, {}
    copies = packed_bytes = 0

    def flush(key):
        nonlocal copies, packed_bytes
        rows = buckets.pop(key, [])
        sizes.pop(key, None)
        if not rows:
            return
        if len(rows) == 1:
            tensor = rows[0]
            arrays[id(tensor)] = tensor.detach().cpu().numpy()
        else:
            # cat preserves bit patterns; no reduction, cast, RNG or autograd.
            packed = torch.cat([x.detach().reshape(-1) for x in rows])
            host = packed.cpu().numpy()
            start = 0
            for tensor in rows:
                count = tensor.numel()
                arrays[id(tensor)] = host[start:start+count].reshape(tuple(tensor.shape))
                start += count
            packed_bytes += packed.numel()*packed.element_size()
        copies += 1

    for tensor in tensors.values():
        # Use the original conversion for unsupported layouts, rather than
        # silently weakening content checks or inventing dtype conversions.
        if tensor.layout != torch.strided or tensor.is_conj() or tensor.is_neg():
            arrays[id(tensor)] = tensor.detach().cpu().numpy()
            copies += 1
            continue
        key = (tensor.device, tensor.dtype)
        size = tensor.numel()*tensor.element_size()
        if key in buckets and sizes[key]+size > max_batch_bytes:
            flush(key)
        buckets.setdefault(key, []).append(tensor)
        sizes[key] = sizes.get(key, 0)+size
        if sizes[key] >= max_batch_bytes:
            flush(key)
    for key in list(buckets):
        flush(key)
    if statistics is not None:
        statistics.update(tensor_occurrences=occurrences, unique_tensors=len(tensors),
                          transfers=copies, packed_bytes=packed_bytes,
                          cross_call_reuse=False, max_batch_bytes=max_batch_bytes)
    return arrays
