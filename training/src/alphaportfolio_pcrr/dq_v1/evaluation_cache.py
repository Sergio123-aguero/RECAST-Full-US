"""Bounded pure-monthly encoding reuse in ONE fixed-weight evaluation only."""
from collections import OrderedDict
import hashlib

import numpy as np
import torch


class MonthlyEvaluationCache:
    def __init__(self, capacity=2):
        if type(capacity) is not int or capacity < 1:
            raise ValueError('positive integer monthly cache capacity required')
        self.capacity = capacity
        self.entries = OrderedDict()
        self.rows = OrderedDict()
        self.max_assets = self.reused_rows = self.encoded_rows = 0
        self.asset_ids = None
        self.token = None
        self.hits = self.misses = self.invalidations = 0

    def bind_asset_ids(self, ids):
        self.asset_ids = np.asarray(ids).copy()

    def resolve(self, model, monthly, observed, compute=None, *, compute_rows=None):
        if torch.is_grad_enabled() or model.training or any(m.training for m in model.monthly_srem.modules()):
            raise ValueError('monthly encoding reuse requires no-gradient fixed-weight evaluation')
        tensors = tuple(model.monthly_srem.parameters())+tuple(model.monthly_srem.buffers())
        token = (model.chunk_size, model.monthly_block_length,
                 tuple((id(t),t._version,t.dtype,t.device) for t in tensors))
        if self.token != token:
            if self.token is not None: self.invalidations += 1
            self.entries.clear()
            self.rows.clear()
            self.token = token
        if compute_rows is not None:
            return self._resolve_rows(model,monthly,observed,compute_rows)
        digest = hashlib.sha256()
        for value in (monthly, observed, self.asset_ids):
            if value is None:
                digest.update(b'no-bound-ids'); continue
            if torch.is_tensor(value): value = value.detach().cpu().numpy()
            value = np.ascontiguousarray(value)
            if value.dtype.hasobject: raise ValueError('numeric monthly cache inputs required')
            digest.update(repr((value.shape,value.dtype.str)).encode())
            digest.update(memoryview(value).cast('B'))
        key = digest.digest()
        if key in self.entries:
            self.hits += 1
            values, versions = self.entries.pop(key)
            if tuple(v._version for v in values) != versions:
                raise RuntimeError('cached monthly context was mutated')
            self.entries[key] = values, versions
            return values
        self.misses += 1
        values = compute()
        if any(v.requires_grad for v in values):
            raise ValueError('monthly evaluation cache must never retain a training graph')
        self.entries[key] = values, tuple(v._version for v in values)
        if len(self.entries)>self.capacity: self.entries.popitem(last=False)
        return values

    def _resolve_rows(self, model, monthly, observed, compute):
        """Asset-independent SREM only; never cache cross-stock/context layers.

        Changing the daily eligible axis must not force a whole-universe miss.
        A miss recomputes the ENTIRE original encoder chunk, not a compacted
        miss-only batch. Keys include that chunk's original batch size. A hit
        gathers independent rows from equal-shape evaluation calls; no padded
        stocks, attention mask or floating-point precision is introduced.
        """
        b,n,t,f = monthly.shape
        count=b*n
        ids = (np.arange(count) if self.asset_ids is None else
               np.tile(self.asset_ids.reshape(-1),b))
        if ids.size != count: raise ValueError('monthly cache asset identity axis mismatch')
        x=monthly.reshape(count,t,f); mask=observed.reshape(count,t,f)
        raw=np.ascontiguousarray(x.detach().cpu().numpy())
        known=np.ascontiguousarray(mask.detach().cpu().numpy())
        self.max_assets=max(self.max_assets,count)
        outputs=[]; reused=0
        for start in range(0,count,model.chunk_size):
            end=min(count,start+model.chunk_size); size=end-start
            keys=[]
            for i in range(start,end):
                digest=hashlib.sha256(repr((int(ids[i]),size,t,f,raw.dtype.str)).encode())
                digest.update(memoryview(raw[i]).cast('B'))
                digest.update(memoryview(known[i]).cast('B'))
                keys.append(digest.digest())
            if all(k in self.rows for k in keys):
                rows=[]
                for key in keys:
                    values,index,versions=self.rows.pop(key)
                    if tuple(v._version for v in values)!=versions:
                        raise RuntimeError('cached monthly rows were mutated')
                    self.rows[key]=(values,index,versions)
                    rows.append(tuple(v[index] for v in values))
                values=tuple(torch.stack([row[j] for row in rows]) for j in range(2))
                reused+=size
            else:
                values=tuple(v.reshape(size,-1) for v in compute(x[start:end][None],mask[start:end][None]))
                if len(values)!=2 or any(v.requires_grad for v in values):
                    raise ValueError('exactly two graph-free monthly block outputs required')
                versions=tuple(v._version for v in values)
                for i,key in enumerate(keys):
                    self.rows.pop(key,None)
                    self.rows[key]=(values,i,versions)
                self.encoded_rows+=size
            outputs.append(values)
        while len(self.rows)>self.capacity*self.max_assets:self.rows.popitem(last=False)
        self.reused_rows+=reused
        if reused==count:self.hits+=1
        else:self.misses+=1
        return tuple(torch.cat([v[j] for v in outputs]).reshape(b,n,-1) for j in range(2))

    def report(self):
        return dict(schema='dq.monthly_evaluation_cache.v2',hits=self.hits,misses=self.misses,
            invalidations=self.invalidations,entries=len(self.entries),capacity=self.capacity,
            row_entries=len(self.rows),row_capacity=self.capacity*self.max_assets,
            reused_rows=self.reused_rows,encoded_rows=self.encoded_rows,
            original_chunk_shape_on_miss=True,asset_independent_encoder_only=True,
            training_reuse=False,cross_evaluation_reuse=False,exact_input_content_binding=True)

    def clear(self):
        self.entries.clear()
        self.rows.clear()
        self.asset_ids = None
