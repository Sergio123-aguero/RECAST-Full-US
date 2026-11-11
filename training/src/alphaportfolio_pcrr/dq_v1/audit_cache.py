"""Ephemeral, input-bound audit constants for one exact full-path replay.

Only no-gradient diagnostics/targets are cached. Predictions, neural graphs,
bootstrap masks and losses always belong to the current forward pass. A cache
is neither checkpointed nor shared across dates, updates, models or attempts.
"""
import copy


def _plain(value):
    if value is None or type(value) in (bool, int, float, str):
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError('audit diagnostic keys must be strings')
            _plain(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _plain(item)
        return
    raise TypeError('audit replay cache accepts plain diagnostics only, never tensors/graphs')


class AuditReplayCache:
    def __init__(self, fingerprint):
        self.fingerprint = fingerprint
        self.entries = {}
        self.used = set()
        self.phase = 'record'
        self.computed = self.hits = 0

    def resolve(self, key, context, compute):
        signature = self.fingerprint(context)
        if self.phase == 'record':
            if key in self.entries:
                raise RuntimeError('duplicate audit record key')
            result = compute()
            _plain(result)
            self.entries[key] = (signature, copy.deepcopy(result))
            self.computed += 1
            return result
        if self.phase != 'replay':
            raise RuntimeError('audit cache used outside its record/replay lifetime')
        if key not in self.entries or key in self.used:
            raise RuntimeError('missing or duplicate audit replay key; optimizer not stepped')
        expected, result = self.entries[key]
        if signature != expected:
            raise RuntimeError('audit input replay mismatch; optimizer not stepped')
        self.used.add(key)
        self.hits += 1
        return copy.deepcopy(result)

    def begin_replay(self):
        if self.phase != 'record':
            raise RuntimeError('audit cache can only replay once')
        self.phase = 'replay'

    def finish_replay(self):
        if self.phase != 'replay' or len(self.used) != len(self.entries):
            raise RuntimeError('incomplete audit replay; optimizer not stepped')
        self.phase = 'closed'

    def report(self):
        return dict(schema='dq.audit_replay_cache.v1', phase=self.phase,
                    computed=self.computed, replay_hits=self.hits,
                    entries=len(self.entries), exact_input_binding=True,
                    cached_tensors=0, cached_graphs=0, cross_update_reuse=False)


def audit_context(search, audit):
    """All numerical/semantic inputs consumed by audit_target, not predictions."""
    return dict(weights=search['weights'], owners=search['owners'],
        previous_weights=search['previous_weights'],
        config=search['config'], round_id=search.get('round_id'),
        exploration_seed=search.get('exploration_seed'), kind=audit['kind'],
        ordinal=audit['ordinal'], base_problem=audit['base_problem'],
        base_amplitudes=audit['base_amplitudes'],
        plan={key: audit['plan'][key] for key in
              ('direction', 'predicted_gain', 'risk_factor', 'slot')})
