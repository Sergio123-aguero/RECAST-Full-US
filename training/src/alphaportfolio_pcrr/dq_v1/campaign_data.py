"""Portable read-only shared six-arm data. Quarterly feature shards, no test IO.

Full-US daily cross-sections are normalized BEFORE projecting onto the fixed
US roster. Exchange subsets only project the shared result. Missing histories
remain masked. Date-only vendor records use explicitly disclosed close-based
research availability assumptions, not claimed historical publication proofs.
"""
from datetime import date, timedelta
import gc
import hashlib
import json
from pathlib import Path
import shutil

import exchange_calendars as xc
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from alphaportfolio_original.data.crsp import CIZ_COMMON_STOCK_FILTERS
from alphaportfolio_original.data.schema import FEATURE_NAMES as MONTHLY_NAMES
from ..daily_multiscale_data import NUMERIC_COLUMNS, _entry_eligible, read_months
from ..daily_terminal_liquidation_v17 import load_terminal_execution
from .feature_expansion import build_daily71
from .feature_contract import feature71_contract, contract_sha256
from .formal71 import prepare_execution71
from .training import PathWindow, aware, fingerprint

DATASETS = ('US_FIXED',)
EXTRA = ('siccd', 'dlyopen', 'dlyclose', 'dlyhigh', 'dlylow', 'dlyprevprc',
         'dlyfacprc', 'dlyorddivamt', 'dlynonorddivamt')
SCHEMA = 'dq.shared_daily71.six_arm_data.v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, obj):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)


def inventory(root):
    root = Path(root)
    return [dict(path=p.relative_to(root).as_posix(), bytes=p.stat().st_size, sha256=sha(p))
            for p in sorted(root.rglob('*')) if p.is_file() and p != root/'MANIFEST.json']


def verify(root, manifest='MANIFEST.json'):
    root = Path(root).resolve()
    meta = read(root/manifest)
    for ref in meta['files']:
        p = (root/ref['path']).resolve()
        if not p.is_relative_to(root) or not p.is_file() or p.stat().st_size != ref['bytes'] or sha(p) != ref['sha256']:
            raise ValueError(f'changed or missing bound file: {p}')
    return meta


def source_proofs(roster_root, monthly_panel, monthly_extension):
    roster_root, monthly_panel, monthly_extension = map(Path, (roster_root, monthly_panel, monthly_extension))
    refs = read(roster_root.parent/'MANIFEST.json')['files']
    checked = []
    def check(p, ref):
        digest = sha(p)
        if p.stat().st_size != ref['bytes'] or digest != ref['sha256']:
            raise ValueError(f'source differs from approved receipt: {p}')
        checked.append(dict(path=str(p), bytes=p.stat().st_size, sha256=digest))
    for ds in DATASETS:
        for suffix in ('.npy', '.parquet'):
            p = roster_root/(ds+suffix)
            check(p, next(r for r in refs if r['path'] == f'rosters/{ds}{suffix}'))
    old = read(monthly_panel.parent/'tier1_full_stock_input_manifest.json')['files']
    check(monthly_panel, next(r for r in old if r['role'] == 'panel'))
    new = {r['file']: r for r in read(monthly_extension.parent/'prepared.json')['feature_files']}
    for month in pd.period_range('2016-01', '2020-11', freq='M'):
        p = monthly_extension/f'{month}.parquet'
        check(p, new[p.name])
    return checked


def raw_rows(raw_root, days, previous):
    """Bound source receipts, common-equity records, exact return-duration mask."""
    frames, proofs = [], []
    columns = list(dict.fromkeys(('permno', 'dlycaldt', *NUMERIC_COLUMNS, *EXTRA,
        *CIZ_COMMON_STOCK_FILTERS, 'dlyprcflg', 'dlyprevdt', 'dlyretmissflg', 'dlyretdurflg', 'dlydelflg')))
    for month in pd.period_range(days[0], days[-1], freq='M'):
        root = Path(raw_root)/'dsf_v2'/str(month)
        receipt = read(root/'receipt.json')
        if receipt['status'] != 'COMPLETE_BOUNDARY_SAFE_PARTITION':
            raise ValueError('incomplete raw partition')
        proofs.append(dict(month=str(month), receipt_sha256=sha(root/'receipt.json'), snapshot_id=receipt['snapshot_id']))
        for part in receipt['parts']:
            p = root/part['file']
            if sha(p) != part['sha256']:
                raise ValueError('raw part checksum mismatch')
            t = pq.read_table(p, columns=columns, filters=[('dlycaldt', '>=', days[0]),
                ('dlycaldt', '<=', days[-1]), ('securitytype', '==', 'EQTY'), ('securitysubtype', '==', 'COM')])
            for name in (*NUMERIC_COLUMNS, *EXTRA):
                t = t.set_column(t.schema.get_field_index(name), name, t[name].cast(pa.float64()))
            f = t.to_pandas(ignore_metadata=True)
            if f.empty:
                continue
            f['dlycaldt'] = pd.to_datetime(f.dlycaldt)
            f['entry_eligible'] = _entry_eligible(f)
            f['traded'] = f.dlyprcflg.eq('TR') & f.tradingstatusflg.eq('A') & f.dlydelflg.eq('N')
            expected = f.dlycaldt.dt.date.map(previous)
            span = (f.dlycaldt-pd.to_datetime(expected)).dt.days
            flags = span.map(lambda v: f'D{int(v)}' if pd.notna(v) and v <= 4 else 'DU')
            valid = (pd.to_datetime(f.dlyprevdt).dt.date.eq(expected) & f.dlyretmissflg.eq('NA')
                & f.dlyretdurflg.eq(flags) & ~f.dlydelflg.eq('Y') & f.dlyret.ge(-1) & np.isfinite(f.dlyret))
            f.loc[~valid, 'dlyret'] = np.nan
            frames.append(f.loc[:, ['permno', 'dlycaldt', 'entry_eligible', 'traded', *NUMERIC_COLUMNS, *EXTRA]])
    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(['permno', 'dlycaldt']).any() or len({p['snapshot_id'] for p in proofs}) != 1:
        raise ValueError('duplicate raw keys or mixed source snapshots')
    return result, proofs


def build_shared(root, *, raw_root, roster_root, monthly_panel, monthly_extension, ledger_root, progress=print):
    """Append-only resumable-by-completed-shard preparation; never trains."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if (root/'READY.json').exists():
        verify(root)
        return read(root/'READY.json')
    source_files = [Path(__file__), Path(__file__).with_name('feature_expansion.py'),
        Path(__file__).parent.parent/'daily_multiscale_data.py',
        Path(__file__).parent.parent/'daily_feature_expansion_v11.py', Path(__file__).with_name('feature_contract.py')]
    lock = dict(schema=SCHEMA, feature_contract=feature71_contract(),
        raw_root=str(raw_root), roster_root=str(roster_root), monthly_panel=str(monthly_panel),
        monthly_extension=str(monthly_extension), ledger_root=str(ledger_root),
        feature_code={p.name: sha(p) for p in source_files}, test_access=False,
        source_proofs=source_proofs(roster_root, monthly_panel, monthly_extension),
        normalization='same_date_full_US_reference_before_fixed_pool_projection',
        availability='exchange_close_plus_1us_inputs_then_plus_2us_decision_research_assumption',
        monthly_missingness='preserve_existing_preprocessed51_values_missing_stock_months_masked',
        retrospective_roster_quality_screen_through='2020-12-31', stocks_newly_deleted=0)
    if (root/'LOCK.json').exists():
        if read(root/'LOCK.json') != lock:
            raise ValueError('partial preparation binding differs; use fresh output')
    else:
        write_new(root/'LOCK.json', lock)
    roster_root, monthly_extension = Path(roster_root), Path(monthly_extension)
    ids = np.load(roster_root/'US_FIXED.npy', allow_pickle=False)
    if not np.array_equal(ids, np.unique(ids)):
        raise ValueError('sorted unique US identities required')
    cal = xc.get_calendar('XNYS', start='2013-12-01', end='2020-12-31')
    all_days = [d.date() for d in cal.sessions]
    prior = {b: a for a, b in zip(all_days, all_days[1:])}
    calendar = {str(d): cal.session_close(pd.Timestamp(d)).isoformat() for d in all_days}
    feature_days = [d for d in all_days if date(2015, 1, 1) <= d <= date(2020, 12, 29)]
    if not (root/'calendar.json').exists():
        write_new(root/'calendar.json', calendar)
        for ds in DATASETS:
            (root/'rosters').mkdir(exist_ok=True)
            for suffix in ('.npy', '.parquet'):
                shutil.copy2(roster_root/(ds+suffix), root/'rosters'/(ds+suffix))
    shard_metas = []
    for quarter in pd.period_range('2015Q1', '2020Q4', freq='Q'):
        output_days = [d for d in feature_days if pd.Period(d, freq='Q') == quarter]
        target = root/'features'/str(quarter)
        if (target/'READY.json').exists():
            verify(target)
            shard_metas.append(read(target/'READY.json'))
            continue
        if target.exists():
            raise ValueError(f'incomplete shard retained for diagnosis; do not overwrite: {target}')
        first_index = all_days.index(output_days[0])
        input_days = all_days[first_index-252:all_days.index(output_days[-1])+1]
        progress(dict(stage='feature_raw', shard=str(quarter), days=len(input_days)))
        rows, proofs = raw_rows(raw_root, input_days, prior)
        active_ids = np.intersect1d(ids, rows.permno.unique())
        # Permanent identities absent from this historical slice are still kept
        # in the fixed cache, with false eligibility and an all-missing history.
        built = build_daily71(rows, input_days, output_start=output_days[0], target_ids=active_ids,
            enforce_memory=True, progress=lambda x: None)
        target.mkdir(parents=True, exist_ok=False)
        values = np.lib.format.open_memmap(target/'values.npy', mode='w+', dtype=np.float32,
            shape=(len(output_days), len(ids), 71))
        values[:] = np.nan
        columns = np.searchsorted(ids, active_ids)
        values[:, columns] = built['values']
        values.flush()
        eligible = np.zeros((len(output_days), len(ids)), bool)
        eligible[:, columns] = built['entry_eligible']
        np.save(target/'eligible.npy', eligible, allow_pickle=False)
        m = dict(shard=str(quarter), dates=[str(d) for d in output_days],
            raw_first=str(input_days[0]), raw_last=str(input_days[-1]), warmup_sessions=252,
            stocks=len(ids), active_ids_present_in_source=len(active_ids),
            reference_assets=built['reference_assets'] if 'reference_assets' in built else int(rows.permno.nunique()),
            feature_contract_sha256=contract_sha256(feature71_contract()), sources=proofs,
            observed_feature_cells=int(np.isfinite(values).sum()), total_feature_cells=int(values.size))
        write_new(target/'READY.json', m)
        write_new(target/'MANIFEST.json', dict(files=inventory(target)))
        shard_metas.append(m)
        del values, eligible, built, rows
        gc.collect()
        progress(dict(stage='feature_complete', shard=str(quarter)))
    if not (root/'monthly').exists():
        dest = root/'monthly'
        dest.mkdir()
        old = read_months(monthly_panel, ['month', 'permno', *MONTHLY_NAMES], '2014-01', '2015-12', period_storage=True)
        old['month'] = pd.PeriodIndex(old.month, freq='M')
        months = pd.period_range('2014-01', '2020-11', freq='M')
        values = np.lib.format.open_memmap(dest/'values.npy', mode='w+', dtype=np.float32,
            shape=(len(months), len(ids), 51))
        sources = [dict(path=str(monthly_panel), sha256=sha(monthly_panel))]
        for i, month in enumerate(months):
            if month < pd.Period('2016-01'):
                f = old.loc[old.month == month]
            else:
                p = monthly_extension/f'{month}.parquet'
                f = pq.read_table(p, columns=['month', 'permno', *MONTHLY_NAMES]).to_pandas()
                sources.append(dict(path=str(p), sha256=sha(p)))
            if f.empty or f.permno.duplicated().any():
                raise ValueError('empty/duplicate monthly partition')
            values[i] = f.set_index('permno').reindex(ids)[list(MONTHLY_NAMES)].to_numpy(np.float32)
        values.flush()
        write_new(dest/'READY.json', dict(months=[str(m) for m in months], sources=sources,
            original_preprocessing_not_refit=True, raw_fundamental_missing_masks_not_reconstructed=True))
        write_new(dest/'MANIFEST.json', dict(files=inventory(dest)))
        del values, old
        gc.collect()
    verify(root/'monthly')
    for ds in DATASETS:
        for role in ('train', 'validation'):
            source = Path(ledger_root)/'execution'/ds/role
            dest = root/'execution'/ds/role
            if not dest.exists():
                dest.mkdir(parents=True)
                for name in ('EXECUTION_V17.json', 'execution_v17.npz', 'MANIFEST.json', 'LIQUIDATION_RECEIPT.json'):
                    shutil.copy2(source/name, dest/name)
            load_terminal_execution(dest)
    ready = dict(schema=SCHEMA, shards=shard_metas,
        stocks={ds: len(np.load(root/'rosters'/f'{ds}.npy', allow_pickle=False)) for ds in DATASETS},
        feature_dates=[str(d) for d in feature_days], feature_contract=feature71_contract(),
        expected_sessions=dict(train=501, validation=754), test_features_included=False,
        full_US_normalization_before_pool_projection=True, source_path_leakage_into_features=False)
    write_new(root/'READY.json', ready)
    write_new(root/'MANIFEST.json', dict(files=inventory(root)))
    return ready


class SharedSource:
    """Read-only mmap and lazy one-date windows. Never loads future outcomes."""
    def __init__(self, root, dataset, *, verify_files=True):
        self.root, self.dataset = Path(root), dataset
        if dataset not in DATASETS:
            raise ValueError('unknown pool')
        self.binding = verify(self.root) if verify_files else read(self.root/'MANIFEST.json')
        self.ready = read(self.root/'READY.json')
        if self.ready['schema'] != SCHEMA or self.ready['feature_contract'] != feature71_contract():
            raise ValueError('shared feature contract mismatch')
        self.ids = np.load(self.root/'rosters/US_FIXED.npy', allow_pickle=False)
        self.roster = np.load(self.root/'rosters'/f'{dataset}.npy', allow_pickle=False)
        if (not np.array_equal(self.ids, np.unique(self.ids))
                or not np.array_equal(self.roster, np.unique(self.roster))
                or not np.isin(self.roster, self.ids).all()):
            raise ValueError('fixed subset identities differ from shared reference')
        self.calendar = read(self.root/'calendar.json')
        self.months = read(self.root/'monthly/READY.json')['months']
        self.monthly = np.load(self.root/'monthly/values.npy', mmap_mode='r', allow_pickle=False)
        self.shards, self.positions, self.dates = [], {}, []
        for meta in self.ready['shards']:
            p = self.root/'features'/meta['shard']
            shard = dict(meta=meta, values=np.load(p/'values.npy', mmap_mode='r', allow_pickle=False),
                eligible=np.load(p/'eligible.npy', mmap_mode='r', allow_pickle=False), path=p)
            for i, day in enumerate(meta['dates']):
                self.positions[day] = (len(self.shards), i)
                self.dates.append(day)
            self.shards.append(shard)
        self.date_indices = {d: i for i, d in enumerate(self.dates)}

    def _daily_window(self, days, columns):
        pieces = []
        for day in days:
            si, ri = self.positions[day]
            pieces.append(self.shards[si]['values'][ri, columns])
        return np.stack(pieces).transpose(1, 0, 2)

    def sample(self, day, target_ids, execution_date):
        day = str(day)
        ids = np.asarray(target_ids)
        if (not np.array_equal(ids, np.unique(ids)) or not np.isin(ids, self.roster).all()
                or len(ids) == 0):
            raise ValueError('nonempty dated target IDs must belong to the fixed pool')
        at = self.date_indices[day]
        if at < 251:
            raise ValueError('short daily history')
        col = np.searchsorted(self.ids, ids)
        x = self._daily_window(self.dates[at-251:at+1], col)
        known = np.isfinite(x)
        cutoff = str(pd.Period(day, freq='M')-1)
        mi = self.months.index(cutoff)
        if mi < 23:
            raise ValueError('short monthly history')
        monthly = self.monthly[mi-23:mi+1][:, col].transpose(1, 0, 2)
        mo = np.isfinite(monthly)
        si, ri = self.positions[day]
        close = aware(self.calendar[day])
        monthly_day = max(d for d in self.calendar if d[:7] == cutoff)
        return dict(signal_date=day, asset_ids=ids.copy(), daily=np.where(known, x, 0).astype(np.float32),
            daily_observed=known, monthly=np.where(mo, monthly, 0).astype(np.float32), monthly_observed=mo,
            daily_feature_names=tuple(feature71_contract()['daily_names']),
            monthly_feature_names=tuple(MONTHLY_NAMES), feature_contract_sha256=contract_sha256(feature71_contract()),
            monthly_cutoff=cutoff, valid_assets=self.shards[si]['eligible'][ri, col].copy(),
            daily_available_at=(close+timedelta(microseconds=1)).isoformat(),
            monthly_available_at=(aware(self.calendar[monthly_day])+timedelta(microseconds=1)).isoformat(),
            decision_at=(close+timedelta(microseconds=2)).isoformat(), execution_at=self.calendar[str(execution_date)])

    def role(self, role, plan):
        if role not in ('train', 'validation'):
            raise ValueError('training bundle never reads test data')
        root = self.root/'execution'/self.dataset/role
        source = load_terminal_execution(root)
        clocks = {str(r['signal_date']): dict(execution_at=self.calendar[str(r['execution_date'])],
            fill_available_at=(aware(self.calendar[str(r['execution_date'])])+timedelta(microseconds=1)).isoformat(),
            outcome_available_at=(aware(self.calendar[str(r['outcome_date'])])+timedelta(microseconds=1)).isoformat())
            for r in source}
        dates = [str(r['signal_date']) for r in source]
        window = PathWindow(role, dates[0], dates[-1], clocks[dates[-1]]['outcome_available_at'])
        rows, ledger, receipt = prepare_execution71(source, window=window, clocks=clocks, plan=plan,
            source_binding=dict(dataset=self.dataset, shared_manifest_sha256=sha(self.root/'MANIFEST.json'),
                execution_npz_sha256=sha(root/'execution_v17.npz'),
                availability_assumption='exchange_close_plus_1us_research_not_verified_publication'))
        if len(rows) != self.ready['expected_sessions'][role]:
            raise ValueError('unexpected full role length')
        by_day = {str(r['signal_date']): r for r in rows}
        def get_sample(day):
            row = by_day[str(day)]
            return self.sample(day, row['target_asset_ids'], row['execution_date'])
        return dates, get_sample, rows, window, ledger, receipt
