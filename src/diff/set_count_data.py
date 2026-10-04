"""Deterministic, pair-disjoint set/count families from the real NRGS pool.

Pool token strings are provenance, not semantic labels in the new contexts.
The builder performs only contextual lexical filtering. Model validation is a
separate step; this manifest never attests to full-completion correctness.
"""
from __future__ import annotations

import argparse
import builtins
import hashlib
import itertools
import json
import keyword
from pathlib import Path
import random

from src.diff import data as D, nrgs_prefix as N
from src.goals.pivot_token_filter import _signature, tokens_differ_meaningfully

VERSION = 'set-count-v2'
RENDERER = 'opaque-identifiers-v2-frontier-b'
SET_RULE = 'source-only-H1'
BLOCK_CONFIGS = {'de': ('core', 'swapped', 'four'), 'da': ('core', 'four')}
BLOCK_DONORS = {'de': ('A', 'B', 'C'), 'da': ('C', 'D')}
SOURCE_REVISION = '8c59544e5d04'
# Per-donor source order: completed tasks first, then b, then the rest. With a fixed 'bade' order the
# model skips b at `=` and fills the next gap (A->e, B->d; Coder-32B, 2026-10-04), which makes "copy the
# source's next task" coincide with the set prediction. Labels are unchanged: H1/H1c depend only on the
# source's completed set / count, not on its order.
SOURCE_ORDERS = {'A': 'adbe', 'B': 'aebd', 'C': 'abde', 'D': 'dbae'}
CONFIGS = {'core': ('ade', 'a'), 'swapped': ('aed', 'a'), 'four': ('adef', 'a')}
DONORS = {'A': 'ad', 'B': 'ae', 'C': 'a', 'D': 'd'}
# One shared permutation assigns every role AND the unfinished variable from
# the same opaque pool. The unfinished name does not encode target or count.
NAME_POOL = ('u', 'v', 'w', 't', 'z', 'q')


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def meaningful_contrast(prefix, left, right):
    """Separate scoped policy: reject a bare-name-only change, not attributes.

    Token fragments are classified after joining the exact prefix. In particular
    s.is + lower/numeric are attributes, and quoted name-looking text is literal.
    This deliberately does not assert semantic equivalence or correctness.
    """
    if not tokens_differ_meaningfully(left, right, prefix=prefix):
        return False
    a, b = _signature(prefix + left, {}), _signature(prefix + right, {})
    if len(a) != len(b):
        return True
    changes = [i for i in range(len(a)) if a[i] != b[i]]
    def bare(seq, i):
        kind, value = seq[i]
        # Builtin operators such as any/len/map/set are operation-bearing,
        # not interchangeable variable bindings. Preserve either-side builtin
        # differences even when the token ends before its opening parenthesis.
        return (kind == 'name' and not keyword.iskeyword(value)
                and not hasattr(builtins, value)
                and not (i and seq[i - 1][1] == '.'))
    return not (changes and all(bare(a, i) and bare(b, i) for i in changes))


def pair_group(p):
    return ':'.join([p.datatype, *sorted((p.b, p.d))])


def split_pairs(pivots, seed):
    """Stratified by datatype; unordered pair is the indivisible split unit."""
    out = {}
    for dt in sorted({p.datatype for p in pivots}):
        keys = sorted({pair_group(p) for p in pivots if p.datatype == dt},
                      key=lambda k: digest([seed, k]))
        n = len(keys)
        n_dev = max(1, round(n * .15)) if n >= 3 else 0
        n_test = max(1, round(n * .15)) if n >= 3 else 0
        for i, k in enumerate(keys):
            out[k] = 'test' if i < n_test else 'dev' if i < n_test + n_dev else 'train'
    return out


def render(roles, names, datatype, order, done, prefix=None):
    arg = N.VAR_NAME[datatype]
    bank = N.BANK[datatype]
    question = D._question([bank[roles[x]].description for x in order], datatype,
                           arg, 'python', 'listed')
    lines = ['<code>', D._signature(datatype, arg, 'python')]
    lines += [D._assign_line(bank[roles[x]], arg, 'python', res_name=names[x])
              for x in order if x in done]
    # Source capture ends exactly at '='; recipient keeps all prefix bytes.
    # Reuse one independent unfinished name across every matched view/reference,
    # never the name of the expected operation and never a semantic 'result' stub.
    lines += [f"    {names['pending']} =" + ('' if prefix is None else ' ' + prefix)]
    return {'question': question, 'code': '\n'.join(lines)}


def make_family(pivot, roles, split, seed, naming_variants=1, *, secondary_pivot):
    """Join two actual binary instruments, preserving the legacy source-only H1.

    de: ABC, native/alternate d/e; da: CD, native/alternate d/a.
    The swapped destination has native e, so it belongs only to de: using a
    d/a instrument there would require an unmeasured third operation.
    """
    q = secondary_pivot
    if (pivot.datatype != q.datatype or (pivot.b, pivot.d) != (roles['d'], roles['e'])
            or (q.b, q.d) != (roles['d'], roles['a'])):
        raise ValueError('readout pivots must match role pairs d/e and d/a')
    if naming_variants < 1:
        raise ValueError('naming_variants must be positive')
    readouts = {
        'de': {'roles': ['d', 'e'], 'prefix': pivot.prefix,
               'pool_tokens': {'d': pivot.b_tok, 'e': pivot.d_tok},
               'pair_group': pair_group(pivot), 'pivot': dict(pivot.__dict__)},
        'da': {'roles': ['d', 'a'], 'prefix': q.prefix,
               'pool_tokens': {'d': q.b_tok, 'a': q.d_tok},
               'pair_group': pair_group(q), 'pivot': dict(q.__dict__)}}
    fid = digest([VERSION, RENDERER, SET_RULE, readouts, roles, seed])[:24]
    family = {'id': fid, 'pair_group': pair_group(pivot),
              'pair_groups': sorted({r['pair_group'] for r in readouts.values()}),
              'split': split, 'datatype': pivot.datatype, 'role_map': roles,
              'prefix': pivot.prefix, 'pool_tokens': readouts['de']['pool_tokens'],
              'readouts': readouts, 'validation_level': 'unvalidated',
              'configs': dict(CONFIGS), 'source_order': dict(SOURCE_ORDERS),
              'donors': dict(DONORS), 'renderer': RENDERER,
              'set_rule': SET_RULE, 'views': []}
    for variant in range(naming_variants):
        rng = random.Random(digest([seed, fid, variant, 'names']))
        vals = list(NAME_POOL); rng.shuffle(vals)
        names = dict(zip(('a', 'b', 'd', 'e', 'f', 'pending'), vals))
        donors = {key: render(roles, names, pivot.datatype, SOURCE_ORDERS[key], done)
                  for key, done in DONORS.items()}
        for block, readout in readouts.items():
            for config in BLOCK_CONFIGS[block]:
                order, done = CONFIGS[config]
                refs = {x: render(roles, names, pivot.datatype, order,
                                 order[:order.index(x)], readout['prefix'])
                        for x in readout['roles']}
                dst = render(roles, names, pivot.datatype, order, done, readout['prefix'])
                for donor in BLOCK_DONORS[block]:
                    src_done = DONORS[donor]
                    pred = N.predict(SOURCE_ORDERS[donor], src_done, order, done)
                    expected = {'set': pred['H1'], 'count': pred['H1c']}
                    if not {pred['H0'], *expected.values()} <= set(readout['roles']):
                        raise ValueError('prediction is outside the binary readout; never relabel it')
                    family['views'].append({
                        'id': f'{fid}/{variant}/{block}/{config}/{donor}',
                        'naming_variant': variant, 'names': names,
                        'block': block, 'config': config, 'donor': donor,
                        'prefix': readout['prefix'], 'readout_roles': readout['roles'],
                        'pool_tokens': readout['pool_tokens'],
                        'diagnostic_pair_group': readout['pair_group'],
                        'src_order': SOURCE_ORDERS[donor], 'src_done': src_done,
                        'dst_order': order, 'dst_done': done, 'source': donors[donor],
                        'destination': dst, 'references': refs, 'native': pred['H0'],
                        'expected': expected})
    return family


def build_dataset(model='Qwen/Qwen2.5-Coder-32B-Instruct', *, seed=0, n_per_pivot=1,
                  datatypes=None, limit=None, naming_variants=1):
    if n_per_pivot < 1 or naming_variants < 1 or (limit is not None and limit < 1):
        raise ValueError('counts and limit must be positive')
    raw = sorted(set(N.load_pivots(model, datatypes=datatypes, core_only=True)),
                 key=lambda p: digest(p.__dict__))
    # Both diagnostic pairs get ONE global split before any join. Reject a
    # cross-split join instead of leaking its secondary pair or merging all
    # connected pairs into a giant train/test component.
    splits = split_pairs(raw, seed)
    usable, rejected = [], []
    for p in raw:
        if meaningful_contrast(p.prefix, p.b_tok, p.d_tok):
            usable.append(p)
        else:
            rejected.append({'pivot': p.__dict__, 'reason': 'contextual_lexical_contrast'})
    index = {}
    for p in usable:
        index.setdefault((p.datatype, p.b, p.d), []).append(p)
    families = []
    for p in usable:
        other = [x for x in sorted(N.BANK[p.datatype]) if x not in (p.b, p.d)]
        choices = []; missing_pair = cross_split = 0
        for a, b, f in itertools.permutations(other, 3):
            if (N._fam(a) in {N._fam(p.b), N._fam(p.d)} or
                N._fam(b) in {N._fam(a), N._fam(p.b), N._fam(p.d)}):
                continue
            matches = index.get((p.datatype, p.b, a), [])
            if not matches:
                missing_pair += 1; continue
            matches = [q for q in matches if splits[pair_group(q)] == splits[pair_group(p)]]
            if not matches:
                cross_split += 1; continue
            roles = dict(a=a, b=b, d=p.b, e=p.d, f=f)
            # One existing secondary pivot per assignment, selected before
            # model validation. No invented prefix or post-hoc token target.
            q = sorted(matches, key=lambda q: digest([seed, roles, q.__dict__]))[0]
            choices.append((roles, q))
        choices.sort(key=lambda rq: digest([seed, p.__dict__, rq[0], rq[1].__dict__]))
        if not choices:
            rejected.append({'pivot': p.__dict__, 'reason': 'no_same_split_secondary_pair',
                             'role_assignments_missing_pair': missing_pair,
                             'role_assignments_cross_split': cross_split})
        for roles, q in choices[:n_per_pivot]:
            families.append(make_family(p, roles, splits[pair_group(p)], seed,
                                        naming_variants, secondary_pivot=q))
    families.sort(key=lambda f: f['id'])
    if limit is not None:
        families = families[:limit]
    pool = Path(N.PIVOT_POOL.format(model=model.split('/')[-1]))
    settings = dict(model=model, seed=seed, n_per_pivot=n_per_pivot,
                    datatypes=sorted(datatypes) if datatypes else None, limit=limit,
                    naming_variants=naming_variants)
    out = {'version': VERSION, 'renderer': RENDERER, 'set_rule': SET_RULE,
           'source_order': dict(SOURCE_ORDERS), 'donors': dict(DONORS),
           'block_configs': BLOCK_CONFIGS, 'block_donors': BLOCK_DONORS,
           'split_policy': 'global-unordered-pair assignment; join same-split pairs only',
           'pair_splits': splits, 'settings': settings, 'source_revision': SOURCE_REVISION,
           'naming_seed': seed, 'source_pool': str(pool),
           'source_pool_sha256': hashlib.sha256(pool.read_bytes()).hexdigest(),
           'renderer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'validation_level': 'unvalidated', 'source_completion_verified': False,
           'full_completion_verified': False, 'families': families, 'rejected': rejected}
    out['fingerprint'] = digest(out)
    return out


def verify_dataset(data):
    copy = dict(data); fp = copy.pop('fingerprint', None)
    if fp != digest(copy):
        raise ValueError('dataset fingerprint mismatch; rebuild instead of reusing stale metadata')
    if data['version'] != VERSION:
        raise ValueError('unsupported dataset schema; rebuild v1 data for the two-block design')
    if (data.get('renderer') != RENDERER or data.get('source_order') != SOURCE_ORDERS
            or data.get('donors') != DONORS or data.get('set_rule') != SET_RULE):
        raise ValueError('dataset design/renderer mismatch; rebuild instead of relabeling old data')
    groups = {}
    for f in data['families']:
        if set(f['pair_groups']) != {r['pair_group'] for r in f['readouts'].values()}:
            raise ValueError('readout pair identity mismatch')
        for group in f['pair_groups']:
            previous = groups.setdefault(group, f['split'])
            if previous != f['split'] or data['pair_splits'].get(group) != f['split']:
                raise ValueError('diagnostic pair leakage across blocks/splits')
        variants = {v['naming_variant'] for v in f['views']}
        expected_conditions = {(block, c, d) for block in BLOCK_CONFIGS
                               for c in BLOCK_CONFIGS[block] for d in BLOCK_DONORS[block]}
        for variant in variants:
            views = [v for v in f['views'] if v['naming_variant'] == variant]
            if len(views) != 13 or {(v['block'], v['config'], v['donor']) for v in views} != expected_conditions:
                raise ValueError('incomplete matched family; never drop a donor or block')
            for v in views:
                r = f['readouts'][v['block']]
                prediction = N.predict(v['src_order'], v['src_done'], v['dst_order'], v['dst_done'])
                if v['expected'] != {'set': prediction['H1'], 'count': prediction['H1c']}:
                    raise ValueError('labels do not match source-only H1/count definitions')
                if (v['src_order'] != SOURCE_ORDERS[v['donor']] or v['src_done'] != DONORS[v['donor']]
                        or v['native'] != prediction['H0']
                        or v['readout_roles'] != r['roles'] or v['prefix'] != r['prefix']
                        or set(v['references']) != set(r['roles'])
                        or not {v['native'], *v['expected'].values()} <= set(r['roles'])):
                    raise ValueError('view/readout identity mismatch')
    return data


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--model', default='Qwen/Qwen2.5-Coder-32B-Instruct')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--n-per-pivot', type=int, default=1)
    p.add_argument('--naming-variants', type=int, default=1)
    p.add_argument('--datatypes', nargs='+', choices=list(N.BANK))
    p.add_argument('--limit', type=int)
    a = p.parse_args(argv); output = a.output; del a.output
    data = build_dataset(**vars(a))
    path = Path(output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps({'families': len(data['families']), 'fingerprint': data['fingerprint'],
                      'rejected': len(data['rejected']), 'output': output}))


if __name__ == '__main__':
    main()
