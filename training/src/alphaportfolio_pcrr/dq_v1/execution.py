"""Decision-local execution metadata, never cached learned representations.

Pinned staging is append-only: an outstanding asynchronous transfer must never
observe a subsequently overwritten CPU buffer. PyTorch's pinned allocator owns
transfer lifetime tracking; all slabs are also retained for this decision.
No parameter, model buffer, RNG, financial policy or checkpoint schema changes.
"""
import torch


class DecisionMetadata:
    def __init__(self, reference, *, enabled=False, slab_elements=65536):
        self.device, self.dtype = reference.device, reference.dtype
        if not isinstance(slab_elements, int) or slab_elements < 1:
            raise ValueError('positive integer slab size required')
        self.enabled = enabled and self.device.type == 'cuda'
        self.slab_elements = slab_elements
        self.slabs = {}
        self.cursors = {}
        self.copies = self.elements = 0

    def tensor(self, values, *, dtype=None):
        dtype = self.dtype if dtype is None else dtype
        if isinstance(values, torch.Tensor):
            raise TypeError('metadata staging accepts host constants, not learned tensors')
        if not self.enabled:
            return torch.tensor(values, device=self.device, dtype=dtype)
        cpu = torch.as_tensor(values, dtype=dtype, device='cpu')
        count = cpu.numel()
        if not count:
            return cpu.to(self.device)
        slabs = self.slabs.setdefault(dtype, [])
        cursor = self.cursors.get(dtype, 0)
        if not slabs or cursor+count > slabs[-1].numel():
            slabs.append(torch.empty(max(count, self.slab_elements), dtype=dtype,
                                     device='cpu', pin_memory=True))
            cursor = 0
        view = slabs[-1][cursor:cursor+count].view(cpu.shape)
        view.copy_(cpu)
        self.cursors[dtype] = cursor+count
        self.copies += 1
        self.elements += count
        return view.to(self.device, non_blocking=True)
