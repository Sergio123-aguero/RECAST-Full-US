"""Decision-local semantic index and shared access-value predictor.

Topology is discrete and immutable within a date. Node contents and stock keys
are differentiable, updated only for touched stocks between macro rounds. IDs
break ties; they are never financial features. No market outcome enters here.
"""
from dataclasses import dataclass
import hashlib

import torch
from torch import nn
from torch.nn import functional as F

from .search import FEEDBACK_FIELDS
from .execution import DecisionMetadata


ACCESS_SCALARS = ('eligible_fraction', 'universe_fraction', 'read_fraction',
    'tree_depth', 'new_coarse_fraction', 'new_fine_fraction', 'repair_amount',
    'remaining_coarse_fraction', 'remaining_fine_fraction')


def context_seed(*parts):
    text = '|'.join(str(int(x)) for x in parts).encode('ascii')
    return int.from_bytes(hashlib.blake2b(text, digest_size=8).digest(), 'little') % (2**63-1)


@dataclass(frozen=True)
class IndexNode:
    indices: torch.Tensor
    children: tuple
    parent: int
    depth: int
    group: int


class SemanticIndex:
    """Learned-summary groups with variance/median balanced subtrees.

    All valid stocks occur in exactly one leaf, including overflow groups.
    An index build still reads the whole universe and is accounted separately.
    """
    def __init__(self, state, keys, groups, valid, ids, *, groups_count, capacity,
                 stage_metadata=False):
        self.state, self.keys, self.ids = state, keys, ids
        self.metadata = DecisionMetadata(state, enabled=stage_metadata)
        self.stage_metadata = stage_metadata
        self.size_tensors = {}
        self.nodes, self.roots = [], []
        self.paths = [[] for _ in range(state.shape[0])]
        self.leaf_of = [-1]*state.shape[0]

        def build(idx, parent, depth, group):
            idx = idx[torch.argsort(ids[idx], stable=True)]
            number = len(self.nodes)
            self.nodes.append(None)
            for i in idx.tolist():
                self.paths[i].append(number)
            if idx.numel() <= capacity:
                children = ()
                for i in idx.tolist():
                    self.leaf_of[i] = number
            else:
                values = keys[idx].detach()
                axis = int(values.var(0, unbiased=False).argmax())
                # Incoming order is canonical ID order. Stable numeric ties
                # retain that order; no stock is dropped to meet the capacity.
                sorted_idx = idx[torch.argsort(values[:, axis], stable=True)]
                split = sorted_idx.numel()//2
                children = (build(sorted_idx[:split], number, depth+1, group),
                            build(sorted_idx[split:], number, depth+1, group))
            canonical = idx[torch.argsort(ids[idx], stable=True)]
            self.nodes[number] = IndexNode(canonical, children, parent, depth, group)
            return number

        for group in range(groups_count):
            idx = torch.where(valid & (groups == group))[0]
            if idx.numel():
                idx = idx[torch.argsort(ids[idx], stable=True)]
                self.roots.append(build(idx, -1, 0, group))
        self.sums = [state[node.indices].sum(0) for node in self.nodes]
        self.key_sums = [keys[node.indices].sum(0) for node in self.nodes]
        self.members = [tuple(node.indices.tolist()) for node in self.nodes]
        self.build_rows = int(valid.sum())
        self.build_memberships = sum(node.indices.numel() for node in self.nodes)
        self.refresh_rows = self.refresh_nodes = 0
        self.version = 0

    def summary(self, node_id):
        count = self.nodes[node_id].indices.numel()
        return self.sums[node_id]/count, F.normalize(self.key_sums[node_id], dim=0)

    def summaries(self, node_ids):
        """Batch node reads: no per-node accelerator normalization kernels."""
        if self.stage_metadata:
            # Membership sizes are immutable, even after a learned-state refresh.
            token = tuple(node_ids)
            if token not in self.size_tensors:
                self.size_tensors[token] = self.metadata.tensor([len(self.members[i]) for i in node_ids])
            sizes = self.size_tensors[token]
        else:
            sizes = self.state.new_tensor([len(self.members[i]) for i in node_ids])
        states = torch.stack([self.sums[i] for i in node_ids])/sizes[:, None]
        keys = F.normalize(torch.stack([self.key_sums[i] for i in node_ids]), dim=-1)
        return states, keys

    def refresh(self, state, touched, key_projection):
        idx = torch.where(touched)[0]
        if not idx.numel():
            self.state = state
            return
        idx = idx[torch.argsort(self.ids[idx], stable=True)]
        new_keys = F.normalize(key_projection(state[idx]), dim=-1)
        delta, key_delta = state[idx]-self.state[idx], new_keys-self.keys[idx]
        affected = {}
        for local, stock in enumerate(idx.tolist()):
            for node in self.paths[stock]:
                affected.setdefault(node, []).append(local)
        for node, positions in affected.items():
            pos = self.metadata.tensor(positions, dtype=torch.long)
            self.sums[node] = self.sums[node]+delta[pos].sum(0)
            self.key_sums[node] = self.key_sums[node]+key_delta[pos].sum(0)
        self.keys = self.keys.index_copy(0, idx, new_keys)
        self.state = state
        self.refresh_rows += idx.numel()
        self.refresh_nodes += len(affected)
        self.version += 1


class AccessValue(nn.Module):
    """Shared node/stock value head; disagreement is a heuristic, not a bound."""
    def __init__(self, dimension, config):
        super().__init__()
        size = 2*dimension+config.key_dim+len(FEEDBACK_FIELDS)+config.qp_rank+len(ACCESS_SCALARS)
        self.trunk = nn.Sequential(nn.Linear(size, dimension), nn.GELU())
        self.heads = nn.ModuleList(nn.Linear(dimension, 1) for _ in range(config.access_heads))
        for head in self.heads:
            nn.init.normal_(head.weight, std=.01)
            nn.init.constant_(head.bias, .01)
        self.gain_scale = config.gain_scale

    def forward(self, features):
        h = self.trunk(features)
        return self.gain_scale*torch.tanh(torch.cat([head(h) for head in self.heads], -1))


def value_features(content, state_summary, query, key_summary, feedback, global_factor,
                   *, eligible, size, total, read, depth, new_coarse, new_fine,
                   amount, coarse_remaining, fine_remaining, config):
    scalar = content.new_tensor([
        eligible/max(size, 1), size/max(total, 1), read/max(size, 1), depth/16.,
        new_coarse/max(config.coarse_budget, 1), new_fine/max(config.fine_edge_budget, 1),
        float(amount.detach()), coarse_remaining/max(config.coarse_budget, 1),
        fine_remaining/max(config.fine_edge_budget, 1)])
    return torch.cat([content, state_summary, query*key_summary, feedback, global_factor, scalar], -1)


@dataclass
class AccessBudget:
    coarse_remaining: int
    nodes_remaining: int
    coarse_used: int = 0
    nodes_used: int = 0
    eligibility_checks: int = 0
    duplicate_reads_avoided: int = 0

    def coarse(self, count):
        if count < 0 or count > self.coarse_remaining:
            raise RuntimeError('coarse decision budget exceeded')
        self.coarse_remaining -= count
        self.coarse_used += count

    def nodes(self, count):
        if count < 0 or count > self.nodes_remaining:
            raise RuntimeError('node decision budget exceeded')
        self.nodes_remaining -= count
        self.nodes_used += count
