"""Two-wave global-budget access planning with protected global exploration.

All actions in a wave use the same solver context. The second wave can open
new groups, not only deepen an old prefix. Exploration buys computation, NOT
holdings. Sparse audits belong to the training loss, never to inference.
"""
import math
import heapq

import torch

from .evidence import FineEvidence
from .hierarchy import context_seed, value_features
from .search import library_problem, solve_library, solver_feedback


class AccessRound:
    def __init__(self, model, *, index, budget, diagnosis, weights, owners,
                 anchors, amounts, eligible, allowance, remaining, round_id,
                 conditioned, base, store, batch, ids, statistics, observed,
                 previous_weights, decision_key, update_step, collect_aux=True):
        self.m, self.c, self.index, self.budget = model, model.config, index, budget
        self.metadata = index.metadata
        requested_lazy = getattr(model, 'lazy_access_auxiliary', False)
        if requested_lazy and not getattr(model, 'batch_access_features', False):
            raise ValueError('lazy auxiliary requires the same batched feature kernel')
        # Deferring a used expression reorders CUDA gradient accumulation even
        # when forward values are exact. Preserve the original autograd graph;
        # dead auxiliary work is elided only in record/evaluation no-grad passes.
        self.lazy_auxiliary = requested_lazy and not torch.is_grad_enabled()
        self.collect_aux = collect_aux
        self.diag, self.w, self.owner = diagnosis, weights, owners
        self.anchors, self.amounts, self.eligible = anchors, amounts, eligible
        self.allowance, self.remaining, self.round_id = allowance, remaining, round_id
        self.previous = previous_weights
        self.ids = ids
        self.queries = torch.nn.functional.normalize(model.query_projection(diagnosis['content']), dim=-1)
        self.evidence = FineEvidence(model, diagnosis, weights, anchors, amounts, conditioned,
            base, store, batch, ids, statistics, observed,
            deferred=getattr(model, 'batch_fine_evidence', False))
        self.evidence.metadata = self.metadata
        self.coarse = [dict() for _ in anchors]
        # Explicit O(K*N) eligibility metadata, transferred once, not repeated
        # device synchronizations for each visited node. IDs only break ties.
        self.id_values = ids.detach().cpu().tolist()
        masks = torch.stack(eligible).detach().cpu().tolist()
        self.unseen = [sorted((i for i, flag in enumerate(mask) if flag), key=self.id_values.__getitem__)
                       for mask in masks]
        self.unseen_positions = [{stock: pos for pos, stock in enumerate(row)} for row in self.unseen]
        self.amount_values = torch.stack(amounts).detach().cpu().tolist()
        self.node_eligible = {}
        self.frontiers = [set(index.roots) for _ in anchors]
        self.events, self.actions, self.audit_candidates = [], [], []
        self.stock_value_evaluations = 0
        self.generator = torch.Generator(device='cpu').manual_seed(context_seed(
            self.c.access_seed, decision_key, update_step, round_id))
        self.seed = self.generator.initial_seed()
        self.start_coarse, self.start_nodes = budget.coarse_used, budget.nodes_used
        self.available = [len(row) for row in self.unseen]
        self.feedback = self.factor = None
        self.node_predictions = {}
        self.heap_frontier = getattr(model, 'heap_access_frontier', False)
        self.frontier_heap, self.prediction_generations = [], {}
        self.prediction_serial = 0
        self.coarse_node_counts = [{} for _ in anchors]
        self.fine_node_counts = [{} for _ in anchors]

    def active(self, slot):
        return bool(self.evidence.order[slot]) or self.evidence.active_slots < self.c.max_plans

    def candidate_values(self, slot, node_id=None):
        raw = self.unseen[slot] if node_id is None else self.eligible_members(slot, node_id)
        keep = [i for i in raw if i not in self.evidence.cache[slot]]
        # Index membership is already in canonical permanent-ID order.
        return keep if self.heap_frontier and node_id is not None else sorted(keep, key=self.id_values.__getitem__)

    def read_count(self, slot, node):
        if self.heap_frontier: return self.coarse_node_counts[slot].get(node,0)
        return sum(i in self.coarse[slot] for i in self.index.members[node])

    def next_choice(self):
        if not self.heap_frontier:
            choices = [(item['priority'], -j, -node, j, node)
                for (j,node),item in self.node_predictions.items()
                if node in self.frontiers[j] and self.active(j) and item['priority']>0]
            return max(choices)[-2:] if choices else None
        while self.frontier_heap:
            _,slot,node,generation = heapq.heappop(self.frontier_heap)
            if (self.prediction_generations.get((slot,node)) == generation
                    and (slot,node) in self.node_predictions
                    and node in self.frontiers[slot] and self.active(slot)):
                return slot,node
        return None

    def candidates(self, slot, node_id=None):
        return self.metadata.tensor(self.candidate_values(slot, node_id), dtype=torch.long)

    def eligible_members(self, slot, node_id):
        if (slot, node_id) not in self.node_eligible:
            members = self.index.members[node_id]
            self.budget.eligibility_checks += len(members)
            # Positions are mutable; membership is fixed at the first read and
            # excludes fine targets separately below. The original mask is fixed.
            positions, known = self.unseen_positions[slot], self.evidence.cache[slot]
            self.node_eligible[slot, node_id] = tuple(i for i in members if i in positions or i in known)
        return self.node_eligible[slot, node_id]

    def remove_unseen(self, slot, stock):
        row, positions = self.unseen[slot], self.unseen_positions[slot]
        position = positions.pop(stock)
        last = row.pop()
        if position < len(row):
            row[position] = last
            positions[last] = position

    def features(self, slot, *, node_id=None, stock=None, count, fine_count):
        if stock is None:
            node = self.index.nodes[node_id]
            summary, key = self.index.summary(node_id)
            size, depth = node.indices.numel(), node.depth
            read = self.read_count(slot,node_id)
            eligible = len(self.eligible_members(slot, node_id))
        else:
            summary, key = self.index.state[stock], self.index.keys[stock]
            size, depth, read, eligible = 1, 0, int(stock in self.coarse[slot]), 1
        return value_features(self.diag['content'][slot], summary, self.queries[slot], key,
            self.feedback[slot], self.factor, eligible=eligible, size=size, total=self.w.numel(),
            read=read, depth=depth, new_coarse=count, new_fine=fine_count,
            amount=self.amounts[slot], coarse_remaining=self.budget.coarse_remaining,
            fine_remaining=self.remaining-self.evidence.evaluations, config=self.c)

    def score_nodes(self, tasks, fine_left):
        tasks = sorted(tasks)
        if not tasks or self.budget.nodes_remaining == 0:
            return
        # Capping metadata work is explicit, separate from reading stock rows.
        tasks = tasks[:self.budget.nodes_remaining]
        scalars, retained = [], []
        for slot, node_id in tasks:
            if not self.active(slot):
                continue
            eligible = self.eligible_members(slot, node_id)
            available = (len(eligible)-self.fine_node_counts[slot].get(node_id,0) if self.heap_frontier
                         else sum(i not in self.evidence.cache[slot] for i in eligible))
            if not available:
                self.frontiers[slot].discard(node_id)
                continue
            quantum = min(self.c.retrieval_depths[1], available, fine_left)
            if not quantum:
                continue
            members = self.index.members[node_id]
            read = self.read_count(slot,node_id)
            count = min(available, self.c.leaf_capacity, self.budget.coarse_remaining)
            size = len(members)
            scalars.append([len(eligible)/size, size/self.w.numel(), read/size,
                self.index.nodes[node_id].depth/16., count/max(self.c.coarse_budget, 1),
                quantum/max(self.c.fine_edge_budget, 1), self.amount_values[slot],
                self.budget.coarse_remaining/max(self.c.coarse_budget, 1),
                (self.remaining-self.evidence.evaluations)/max(self.c.fine_edge_budget, 1)])
            retained.append((slot, node_id, count, quantum))
        if not retained:
            return
        slots = self.metadata.tensor([row[0] for row in retained], dtype=torch.long)
        means, keys = self.index.summaries([row[1] for row in retained])
        features = torch.cat([self.diag['content'][slots], means, self.queries[slots]*keys,
            self.feedback[slots], self.factor.expand(len(retained), -1), self.metadata.tensor(scalars)], -1)
        prediction = self.m.access_value(features)
        self.budget.nodes(len(retained))
        moments = torch.stack([prediction.mean(-1), prediction.std(-1, unbiased=False)], -1).detach().cpu().tolist()
        for task, value, (mean, std) in zip(retained, prediction, moments):
            slot, node_id, count, quantum = task
            cost = count+self.c.access_fine_cost*quantum+self.c.access_solve_cost
            self.node_predictions[slot, node_id] = dict(priority=(mean+self.c.access_uncertainty*std)/max(cost, 1e-8),
                prediction=value, mean_value=mean, fine_count=quantum)
            if self.heap_frontier:
                self.prediction_serial += 1
                self.prediction_generations[slot,node_id] = self.prediction_serial
                priority = self.node_predictions[slot,node_id]['priority']
                if priority>0:
                    heapq.heappush(self.frontier_heap,(-priority,slot,node_id,self.prediction_serial))

    def ancestor_scalars(self, slot, path, count, fine_count):
        """Snapshot ONLY pre-access host metadata; counters mutate after access."""
        scalars = []
        for node_id in path:
            members = self.index.members[node_id]
            size = len(members)
            read = self.read_count(slot,node_id)
            eligible = len(self.eligible_members(slot, node_id))
            scalars.append([eligible/max(size, 1), size/max(self.w.numel(), 1), read/max(size, 1),
                self.index.nodes[node_id].depth/16., count/max(self.c.coarse_budget, 1),
                fine_count/max(self.c.fine_edge_budget, 1), self.amount_values[slot],
                self.budget.coarse_remaining/max(self.c.coarse_budget, 1),
                (self.remaining-self.evidence.evaluations)/max(self.c.fine_edge_budget, 1)])
        return scalars

    def ancestor_features(self, slot, path, count, fine_count, *, snapshot=None):
        """Read the same pre-access summaries in a single tensor batch."""
        if snapshot is None and not (getattr(self.m, 'batch_fine_evidence', False)
                or getattr(self.m, 'batch_access_features', False)):
            return torch.stack([self.features(slot, node_id=p, count=count, fine_count=fine_count)
                                for p in path])
        scalars, feedback, factor = (snapshot if snapshot is not None else
            (self.ancestor_scalars(slot, path, count, fine_count), self.feedback[slot], self.factor))
        summaries, keys = self.index.summaries(path)
        return torch.cat([self.diag['content'][slot].expand(len(path), -1), summaries,
            self.queries[slot]*keys, feedback.expand(len(path), -1),
            factor.expand(len(path), -1), self.metadata.tensor(scalars)], -1)

    def inspect(self, slot, idx, fine_count, *, explore, node_id, wave, base_problem, base_solution,
                index_values=None):
        """Compute new coarse rows, select targets, then protect fine exploration."""
        index_values = idx.tolist() if index_values is None else index_values
        fresh = [i for i in index_values if i not in self.coarse[slot]]
        if len(fresh) > self.budget.coarse_remaining:
            raise RuntimeError('planner proposed unfunded coarse rows')
        # Train the value of an access action using its PRE-access features.
        # No newly read rows, fine tensors or consumed budget enter this head.
        path = self.index.paths[index_values[0]] if explore else self._ancestors(node_id)
        path = path[:self.budget.nodes_remaining]
        path_predictions = []
        deferred_context = None
        if path:
            if self.lazy_auxiliary:
                scalars = self.ancestor_scalars(slot,path,len(fresh),min(fine_count,idx.numel()))
                if self.collect_aux:
                    # Index contents, query and source book are frozen within
                    # this AccessRound. Preserve wave-specific feedback and the
                    # PRE-access counters, not their values at audit selection.
                    deferred_context = (tuple(path), self.index.version,
                        scalars,
                        self.feedback[slot], self.factor)
            else:
                features = self.ancestor_features(slot, path, len(fresh), min(fine_count, idx.numel()))
                path_predictions = list(self.m.access_value(features).unbind())
            # Logical admission/stop budget stays identical even if an unused
            # auxiliary prediction is not physically evaluated.
            self.budget.nodes(len(path))
        if fresh:
            rows = (idx if self.index.stage_metadata and fresh == index_values else
                    self.metadata.tensor(fresh, dtype=torch.long))
            dots = self.index.keys[rows] @ self.queries[slot]
            self.coarse[slot].update(zip(fresh, dots.unbind()))
            if self.heap_frontier:
                for stock in fresh:
                    for node in self.index.paths[stock]:
                        counts = self.coarse_node_counts[slot]
                        counts[node] = counts.get(node,0)+1
            self.budget.coarse(len(fresh))
        self.budget.duplicate_reads_avoided += idx.numel()-len(fresh)
        count = idx.numel()
        scalar = self.metadata.tensor([1., 1/self.w.numel(), 1., 0., 0.,
            1/max(self.c.fine_edge_budget, 1), self.amount_values[slot],
            self.budget.coarse_remaining/max(self.c.coarse_budget, 1),
            (self.remaining-self.evidence.evaluations)/max(self.c.fine_edge_budget, 1)])
        stock_features = torch.cat([self.diag['content'][slot].expand(count, -1), self.index.state[idx],
            self.queries[slot]*self.index.keys[idx], self.feedback[slot].expand(count, -1),
            self.factor.expand(count, -1), scalar.expand(count, -1)], -1)
        stock_predictions = self.m.access_value(stock_features)
        self.stock_value_evaluations += idx.numel()
        coarse = torch.stack([self.coarse[slot][i] for i in index_values])
        coarse = coarse+stock_predictions.mean(-1)/self.c.gain_scale
        order = torch.arange(idx.numel(), device=idx.device) if explore else torch.argsort(coarse, descending=True, stable=True)
        selected = order[:fine_count]
        targets = idx[selected]
        target_values = (index_values[:fine_count] if explore else
                         [index_values[i] for i in selected.detach().cpu().tolist()])
        # The sampled target is not filtered through a greedy coarse top-k.
        added = self.evidence.add(slot, targets, coarse[selected], stock_predictions[selected],
                                  stock_indices=target_values)
        if not added:
            return
        for stock in added:
            self.remove_unseen(slot, stock)
            if self.heap_frontier:
                for node in self.index.paths[stock]:
                    counts = self.fine_node_counts[slot]
                    counts[node] = counts.get(node,0)+1
        # Targets refer only to this actual bounded quantum, never to unread
        # members of the ancestor groups or a sum of individual stock values.
        action = dict(slot=slot, node=node_id, wave=wave, exploration=explore,
            fine_ids=[self.id_values[i] for i in target_values], fine_count=len(added),
            coarse_count=len(fresh), ancestor_ids=path)
        self.actions.append(action)
        self.audit_candidates.append(dict(kind='node', predictions=path_predictions,
            prediction_count=len(path), deferred_context=deferred_context,
            slot=slot, indices=tuple(added), base_problem=base_problem, base_amplitudes=base_solution.amplitudes.detach(),
            action=action, prior_library_count=base_problem['gains'].numel()))
        # A stock audit uses the same context and already computed fine tensors.
        stock = added[int(torch.randint(len(added), (), generator=self.generator))]
        self.audit_candidates.append(dict(kind='stock',
            predictions=[self.evidence.cache[slot][stock]['prediction']],
            slot=slot, indices=(stock,), base_problem=base_problem,
            base_amplitudes=base_solution.amplitudes.detach(), action=action,
            prior_library_count=base_problem['gains'].numel()))

    def _ancestors(self, node_id):
        path = []
        while node_id >= 0:
            path.append(node_id)
            node_id = self.index.nodes[node_id].parent
        return list(reversed(path))

    def explore(self, count, *, wave, base_problem, base_solution):
        for _ in range(count):
            pools = [(j, self.unseen[j] if self.budget.coarse_remaining > 0 else
                      [i for i in self.coarse[j] if i in self.unseen_positions[j]])
                     for j in range(len(self.anchors)) if self.active(j)]
            pools = [(j, row) for j, row in pools if row]
            if not pools:
                break
            jpos = int(torch.randint(len(pools), (), generator=self.generator))
            slot, row = pools[jpos]
            pick = int(torch.randint(len(row), (), generator=self.generator))
            stock_value = row[pick]
            stock = self.metadata.tensor([stock_value], dtype=torch.long)
            self.events.append(dict(wave=wave, slot=slot, asset_id=self.id_values[row[pick]],
                slot_probability=1/len(pools), stock_probability_given_slot=1/len(row),
                probability_scope='sequential_draw_conditional_on_current_eligible_unseen_set',
                global_support_available=self.budget.coarse_remaining > 0))
            self.inspect(slot, stock, 1, explore=True, node_id=self.index.leaf_of[stock_value],
                wave=wave, base_problem=base_problem, base_solution=base_solution, index_values=[stock_value])

    def wave(self, limit, *, number, base_problem, base_solution):
        start = self.evidence.evaluations
        reserve = min(limit, max(1, math.ceil(limit*self.c.access_exploration))) if limit else 0
        self.explore(reserve, wave=number, base_problem=base_problem, base_solution=base_solution)
        self.node_predictions = {}
        self.frontier_heap.clear()
        self.prediction_generations.clear()
        self.score_nodes([(j, node) for j, frontier in enumerate(self.frontiers) for node in frontier],
                         limit-(self.evidence.evaluations-start))
        while self.evidence.evaluations-start < limit:
            choice = self.next_choice()
            if choice is None:
                break
            slot, node_id = choice
            self.frontiers[slot].discard(node_id)
            node = self.index.nodes[node_id]
            left = limit-(self.evidence.evaluations-start)
            if node.children:
                self.frontiers[slot].update(node.children)
                self.score_nodes([(slot, child) for child in node.children], left)
                continue
            values = self.candidate_values(slot, node_id)
            if not values:
                continue
            cached = [i for i in values if i in self.coarse[slot]]
            fresh = [i for i in values if i not in self.coarse[slot]]
            take = min(len(fresh), self.budget.coarse_remaining)
            if take < len(fresh):
                permutation = torch.randperm(len(fresh), generator=self.generator)[:take].tolist()
                fresh = [fresh[k] for k in permutation]
            chosen = cached+fresh
            if not chosen:
                continue
            chosen.sort(key=self.id_values.__getitem__)
            idx = self.metadata.tensor(chosen, dtype=torch.long)
            fine_count = min(left, self.c.retrieval_depths[1], idx.numel())
            self.inspect(slot, idx, fine_count, explore=False, node_id=node_id,
                wave=number, base_problem=base_problem, base_solution=base_solution, index_values=chosen)
            # Reopening this leaf in the next wave can inspect remaining stocks;
            # within this wave its action budget is a single bounded quantum.
            if self.candidate_values(slot, node_id):
                self.frontiers[slot].add(node_id)
            self.node_predictions.pop((slot, node_id), None)

    def run(self):
        empty = library_problem(self.w, [], self.c)
        no_op = solve_library(self.w, self.owner, empty, self.c, previous_weights=self.previous)
        self.feedback, self.factor = solver_feedback(self.w, self.owner, self.anchors, self.amounts,
            [0]*len(self.anchors), self.available, empty, no_op, self.c, self.remaining, self.previous)
        first_limit = min(self.allowance, max(1, self.allowance//2)) if self.allowance else 0
        self.wave(first_limit, number=0, base_problem=empty, base_solution=no_op)
        records = [self.evidence.plan(j) for j, items in enumerate(self.evidence.order) if items]
        base_counts = [len(x) for x in self.evidence.order]
        problem = library_problem(self.w, records, self.c)
        probe = solve_library(self.w, self.owner, problem, self.c, previous_weights=self.previous)
        self.feedback, self.factor = solver_feedback(self.w, self.owner, self.anchors, self.amounts,
            base_counts, self.available, problem, probe, self.c,
            self.remaining-self.evidence.evaluations, self.previous)
        probe_feedback = self.feedback.detach().cpu().tolist()
        base_count = len(records)
        self.wave(self.allowance-self.evidence.evaluations, number=1, base_problem=problem, base_solution=probe)
        records += [self.evidence.plan(j) for j, items in enumerate(self.evidence.order) if len(items) > base_counts[j]]
        expanded = len(records) > base_count
        final_problem = library_problem(self.w, records, self.c, base_problem=problem) if expanded else problem
        solution = (solve_library(self.w, self.owner, final_problem, self.c, previous_weights=self.previous,
                                 incumbent=probe.amplitudes) if expanded else probe)
        depths = [len(x) for x in self.evidence.order]
        final_feedback, self.factor = solver_feedback(self.w, self.owner, self.anchors, self.amounts,
            depths, self.available, final_problem, solution, self.c,
            self.remaining-self.evidence.evaluations, self.previous)
        self.feedback = final_feedback
        self.node_predictions = {}
        # Budgeted feedback recheck for stopping; no extra stock/fine/solver work.
        self.score_nodes([(j, node) for j, frontier in enumerate(self.frontiers) for node in frontier],
                         max(0, self.remaining-self.evidence.evaluations))
        opportunity = sum(max([0.]+[x['mean_value'] for (slot, _), x in
                                   self.node_predictions.items() if slot == j]) for j in range(len(depths)))
        unknown = self.budget.nodes_remaining == 0 and any(self.frontiers)
        if unknown:
            opportunity = max(opportunity, self.c.gain_scale)
        # Balanced sampled audit types; inspection probabilities are not passed
        # off as final-set inclusion probabilities or unbiased utility weights.
        audits = []
        for ordinal in range(self.c.counterfactual_queries):
            kind = 'node' if ordinal % 2 == 0 else 'stock'
            candidates = [a for a in self.audit_candidates if a['kind'] == kind
                          and a.get('prediction_count', len(a['predictions']))]
            if candidates:
                selected = int(torch.randint(len(candidates), (), generator=self.generator))
                candidate = candidates[selected]
                chosen = dict(candidate)
                context = chosen.pop('deferred_context', None)
                chosen.pop('prediction_count', None)
                if context is not None:
                    path, version, scalars, feedback, factor = context
                    if version != self.index.version:
                        raise RuntimeError('deferred auxiliary context crossed an index refresh')
                    features = self.ancestor_features(chosen['slot'], path, 0, 0,
                        snapshot=(scalars, feedback, factor))
                    chosen['predictions'] = list(self.m.access_value(features).unbind())
                    # Sampling is with replacement. Repeated draws must share
                    # the original expression and its gradient accumulation,
                    # not recompute a distinct copy of the same prediction.
                    candidate['predictions'] = chosen['predictions']
                    candidate['deferred_context'] = None
                # Build only sampled counterfactual plans from immutable cached
                # evidence, not two full length-N plans for EVERY inspected action.
                audits.append(dict(chosen, plan=self.evidence.plan(chosen['slot'], chosen['indices']),
                                   audit_draw_probability=1/len(candidates), ordinal=ordinal))
        trace = dict(policy='hierarchical_v1', index_version=self.index.version,
            main_solver_calls=1+int(expanded), plan_versions=len(records),
            unique_plan_slots=len({r['slot'] for r in records}),
            max_versions_per_slot=max([0]+[sum(r['slot'] == j for r in records) for j in range(len(depths))]),
            fine_edges=self.evidence.evaluations, fine_cache_hits=self.evidence.hits,
            cached_fine_rows_read_for_plans=self.evidence.plan_reads,
            coarse_rows=self.budget.coarse_used-self.start_coarse,
            node_value_evaluations=self.budget.nodes_used-self.start_nodes,
            stock_value_evaluations=self.stock_value_evaluations,
            coarse_budget_remaining=self.budget.coarse_remaining, node_budget_remaining=self.budget.nodes_remaining,
            exploration_seed=self.seed, exploration_events=self.events, access_actions=self.actions,
            probe_solver=probe.diagnostics, probe_feedback=probe_feedback,
            final_feedback=final_feedback.detach().cpu().tolist(),
            remaining_opportunity_proxy=opportunity, opportunity_unknown=unknown,
            solver_feedback_reopens_unvisited_groups=True, source_book_frozen_within_round=True,
            queries_frozen_within_round=True, incumbent_objective_comparable=True,
            cross_round_fine_cache_reused=False)
        return dict(records=records, problem=final_problem, solution=solution, depths=depths,
            spent=self.evidence.evaluations, opportunity=opportunity, trace=trace,
            auxiliary=dict(weights=self.w, owners=self.owner, previous_weights=self.previous,
                audits=audits, config=self.c.to_dict(), round_id=self.round_id, exploration_seed=self.seed))
