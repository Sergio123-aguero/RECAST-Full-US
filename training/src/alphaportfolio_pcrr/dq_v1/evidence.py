"""Append-only slot/stock fine evidence under one frozen decision context."""
import math

import torch


class FineEvidence:
    def __init__(self, model, diagnosis, weights, anchors, amounts, conditioned,
                 base, store, batch, ids, statistics, observed, *, deferred=False):
        self.model, self.diag, self.w = model, diagnosis, weights
        self.anchors, self.amounts = anchors, amounts
        self.conditioned, self.base, self.store, self.batch = conditioned, base, store, batch
        self.ids, self.statistics, self.observed = ids, statistics, observed
        self.cache = [dict() for _ in anchors]
        self.order = [[] for _ in anchors]
        self.active_slots = 0  # append-only membership, never a model/score cache
        n = weights.numel()
        self.holding = torch.stack([torch.log1p(n*weights.clamp_min(0)),
            torch.log1p(n*(-weights).clamp_min(0)), (weights > 0).to(weights.dtype),
            (weights < 0).to(weights.dtype)], -1)
        self.evaluations = self.hits = self.plan_reads = 0
        self.deferred = deferred
        self.pending = []
        self.fine_batches = 0
        self.slot_queries = {}

    def index_tensor(self, values):
        metadata = getattr(self, 'metadata', None)
        return (torch.tensor(values, device=self.w.device, dtype=torch.long) if metadata is None
                else metadata.tensor(values, dtype=torch.long))

    def slot_query(self, slot):
        reuse = getattr(self.model, 'reuse_fine_queries', False)
        if reuse and slot in self.slot_queries:
            return self.slot_queries[slot]
        source = self.anchors[slot] @ self.conditioned
        query = self.model.fine_query(torch.cat([self.diag['content'][slot], source]))
        if reuse:
            # Differentiable context is frozen for this round, not detached or
            # persisted across rounds/dates. Original vector kernel is retained.
            self.slot_queries[slot] = (source, query)
        return source, query

    def add(self, slot, indices, coarse, predictions, *, stock_indices=None):
        m, c, d = self.model, self.model.config, self.model.dimension
        added, positions, seen = [], [], set(self.cache[slot])
        stocks = indices.tolist() if stock_indices is None else stock_indices
        if len(stocks) != indices.numel():
            raise ValueError('fine evidence CPU index length mismatch')
        for pos, stock in enumerate(stocks):
            if stock in seen:
                self.hits += 1
            else:
                seen.add(stock)
                added.append(stock)
                positions.append(pos)
        if not added:
            return []
        newly_active = not self.order[slot]
        # AccessRound supplies unique still-unseen targets and an exact CPU
        # mirror. Reuse these funded GPU rows instead of copying their indices
        # CPU->GPU again and gathering an identity permutation. Duplicate/mixed
        # cache hits retain the original filtering path.
        if getattr(m, 'reuse_fine_inputs', False) and len(added) == indices.numel():
            idx = indices
            selected_coarse, selected_predictions = coarse, predictions
        else:
            idx = self.index_tensor(added)
            pos = self.index_tensor(positions)
            selected_coarse, selected_predictions = coarse[pos], predictions[pos]
        if self.deferred:
            # Membership, action budgets, stored stock predictions and RNG
            # advance immediately, exactly as in the scalar implementation.
            # Fine logits/values cannot influence selection inside this wave.
            for j, stock in enumerate(added):
                self.cache[slot][stock] = dict(coarse=selected_coarse[j],
                                              prediction=selected_predictions[j])
                self.order[slot].append(stock)
            self.pending.append((slot, added, selected_coarse))
            self.evaluations += len(added)
            self.active_slots += int(newly_active)
            return added
        source, query = self.slot_query(slot)
        correction = self.conditioned[idx]-self.base[idx]
        key = self.store['fine_key'][self.batch, idx]+m.fine_key(correction)+m.holding_key(self.holding[idx])
        values = self.store['fine_value'][self.batch, idx]+m.fine_value(correction)+m.holding_value(self.holding[idx])
        local = torch.cat([self.diag['content'][slot].expand(len(added), d),
            self.conditioned[idx]-source, self.holding[idx], self.statistics[idx],
            self.observed[idx].to(self.w.dtype)], -1)
        logits = key @ query/math.sqrt(c.key_dim)+selected_coarse+m.marginal_score(local)[:, 0]
        for j, stock in enumerate(added):
            self.cache[slot][stock] = dict(logit=logits[j], value=values[j],
                coarse=selected_coarse[j], prediction=selected_predictions[j])
            self.order[slot].append(stock)
        self.evaluations += len(added)
        self.active_slots += int(newly_active)
        self.fine_batches += 1
        return added

    def flush(self):
        """Compute only the already funded rows, once per frozen-context wave.

        Batched GEMM can change last-bit rounding, not the model or objective.
        No RNG is consumed and no unread/unfunded relation is computed here.
        """
        if not self.pending:
            return
        m, c = self.model, self.model.config
        slots = sorted({slot for slot, _, _ in self.pending})
        lookup = {slot: i for i, slot in enumerate(slots)}
        source = torch.stack([self.anchors[slot] @ self.conditioned for slot in slots])
        slot_tensor = self.index_tensor(slots)
        contents = self.diag['content'][slot_tensor]
        # Keep the original vector projection/reduction shape for sensitive
        # scores. Share it per slot rather than recomputing per access action.
        queries = torch.stack([m.fine_query(torch.cat([contents[i], source[i]]))
                               for i in range(len(slots))])
        pairs = [(slot, stock) for slot, stocks, _ in self.pending for stock in stocks]
        idx = self.index_tensor([stock for _, stock in pairs])
        row = self.index_tensor([lookup[slot] for slot, _ in pairs])
        correction = self.conditioned[idx]-self.base[idx]
        holding = self.holding[idx]
        key = self.store['fine_key'][self.batch, idx]+m.fine_key(correction)+m.holding_key(holding)
        values = self.store['fine_value'][self.batch, idx]+m.fine_value(correction)+m.holding_value(holding)
        local = torch.cat([contents[row], self.conditioned[idx]-source[row], holding,
                          self.statistics[idx], self.observed[idx].to(self.w.dtype)], -1)
        marginal = m.marginal_score(local)[:, 0]
        # Preserve each original action's dot-product shape. The expensive
        # key/value/MLP projections remain batched across the funded wave.
        pieces, start = [], 0
        for slot, stocks, coarse in self.pending:
            end = start+len(stocks)
            pieces.append(key[start:end] @ queries[lookup[slot]]/math.sqrt(c.key_dim)
                          +coarse+marginal[start:end])
            start = end
        logits = torch.cat(pieces)
        for j, (slot, stock) in enumerate(pairs):
            self.cache[slot][stock].update(logit=logits[j], value=values[j])
        self.pending.clear()
        self.fine_batches += 1

    def plan(self, slot, indices=None):
        self.flush()
        m, c = self.model, self.model.config
        stocks = self.order[slot] if indices is None else list(indices)
        if not stocks or len(stocks) != len(set(stocks)):
            raise ValueError('a plan needs unique, already computed fine evidence')
        self.plan_reads += len(stocks)
        rows = [self.cache[slot][i] for i in stocks]
        idx = self.index_tensor(stocks)
        logits = torch.stack([r['logit'] for r in rows])
        coarse = torch.stack([r['coarse'] for r in rows])
        values = torch.stack([r['value'] for r in rows])
        probabilities = torch.softmax(logits, 0)
        anchor, amount = self.anchors[slot], self.amounts[slot]
        source = anchor @ self.conditioned
        pool = probabilities @ values
        target = torch.zeros_like(self.w).scatter(0, idx, probabilities)
        features = torch.cat([self.diag['content'][slot], pool, source])
        gain = amount*c.gain_scale*torch.tanh(m.plan_gain(features)[0])*torch.sigmoid(self.diag['intent'][slot])
        return dict(slot=slot, depth=len(stocks), indices=idx, asset_ids=self.ids[idx],
            anchor=anchor, amount=amount, sign=self.diag['signs'][slot], logits=logits,
            coarse_logits=coarse, probabilities=probabilities,
            direction=self.diag['signs'][slot]*amount*(target-anchor), predicted_gain=gain,
            risk_factor=amount*m.plan_risk(torch.cat([pool, source])),
            diagnostic_content=self.diag['content'][slot], source_hidden=source, value_pool=pool)
