"""Small dependency-free, testable policies for the bounded calibration smoke."""
from collections import defaultdict
import math

DATATYPES = ('str', 'int', 'list[int]')
CASES = (('set', 'head'), ('count', 'head'), ('set', 'svd'), ('count', 'svd'))
REVISION = '381fc969f78efac66bc87ff7ddeadb7e73c218a7'
MODEL = 'Qwen/Qwen2.5-Coder-32B-Instruct'


def parse_layers(text):
    if '..' in text:
        lo, hi = map(int, text.split('..'))
        layers = list(range(lo, hi + 1))
    else:
        layers = [int(x) for x in text.split(',')]
    if not layers or len(set(layers)) != len(layers) or min(layers) < 0 or max(layers) >= 64:
        raise ValueError('layers must be unique Qwen2.5-32B layer indices in [0,63]')
    return layers


def select_first(candidates, validate, on_attempt=None):
    """Predetermined datatype/candidate order; never select on optimization outcomes."""
    selected, rejected, attempts = [], [], []
    for datatype in DATATYPES:
        for family in candidates[datatype]:
            accepted, failures = validate([family])
            rejected.extend(failures)
            record = {'datatype': datatype, 'family_id': family['id'],
                      'accepted': bool(accepted), 'rejections': failures}
            attempts.append(record)
            if on_attempt is not None:
                on_attempt(family, accepted, record, attempts, rejected)
            if accepted:
                if len(accepted) != 1 or accepted[0][0]['id'] != family['id']:
                    raise ValueError('validator changed family identity')
                selected.extend(accepted)
                break
    return selected, rejected, attempts


def aggregate(rows):
    """Equal-family means over the fixed nine-view collections, not cycling batches.

    Split means are normalized within native-control/primary subsets. Their
    contributions sum to total CE, making control-only improvement visible.
    """
    if not rows:
        raise ValueError('cannot aggregate zero views')
    groups = defaultdict(list)
    for row in rows:
        groups[row['family_id']].append(row)
    weighted = [(r, 1 / (len(groups) * len(group)))
                for group in groups.values() for r in group]
    def mean(field, subset):
        pairs = [(r, w) for r, w in weighted if subset(r)]
        mass = sum(w for _, w in pairs)
        return sum(w * r[field] for r, w in pairs) / mass if mass else None
    all_rows = lambda r: True
    control = lambda r: r['expected'] == r['native']
    primary = lambda r: not control(r)
    result = {'n_families': len(groups), 'n_views': len(rows),
              'total_expected_ce': mean('expected_ce', all_rows),
              'native_expected_control_ce': mean('expected_ce', control),
              'nonnative_primary_ce': mean('expected_ce', primary),
              'control_ce_contribution': sum(w * r['expected_ce'] for r, w in weighted if control(r)),
              'primary_ce_contribution': sum(w * r['expected_ce'] for r, w in weighted if primary(r)),
              'n_control_views': sum(control(r) for r in rows),
              'n_primary_views': sum(primary(r) for r in rows),
              'whole_family_accuracy': sum(all(r['expected_correct'] for r in g) for g in groups.values()) / len(groups)}
    for name, subset in [('all', all_rows), ('control', control), ('primary', primary)]:
        for field in ('expected_correct', 'native_correct', 'target_correct',
                      'expected_logodds', 'target_native_logodds'):
            metric = field.replace('_correct', '_accuracy')
            result[f'{name}/{metric}'] = mean(field, subset)
    if any(not math.isfinite(v) for v in result.values() if isinstance(v, (int, float))):
        raise ValueError('nonfinite evaluation metric')
    return result


def signed_changes(initial, final):
    return {key: final[key] - value for key, value in initial.items()
            if isinstance(value, (float, int)) and not key.startswith('n_')
            and isinstance(final.get(key), (float, int))}
