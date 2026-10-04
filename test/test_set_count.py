"""Tiny CPU regression checks, no downloads and no pretrained claims."""
import copy
import json
import pytest
import torch

from src.diff import nrgs_prefix as N
from src.diff.set_count_data import (build_dataset, digest, make_family,
                                     meaningful_contrast, split_pairs, verify_dataset,
                                     VERSION, RENDERER, SOURCE_ORDERS, DONORS, NAME_POOL, SET_RULE)
from src.diff.set_count_train import (ByteTokenizer, HFAdapter, chat_prompt, expected_loss,
                                     load_checkpoint, patched, save_checkpoint, train_step,
                                     validate_and_cache, evaluate, view_weights)
from src.diff.query_mask_ops import svd_bases


@pytest.fixture(scope='module', autouse=True)
def bounded_threads():
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def family():
    p = N.Pivot('int', 'is_even', 'is_divisible_by_three', 'n % ', '2', '3')
    roles = dict(a='is_positive', b='is_perfect_square', d='is_even',
                 e='is_divisible_by_three', f='is_power_of_two')
    return make_family(p, roles, 'train', 17, naming_variants=2,
                       secondary_pivot=N.Pivot('int', 'is_even', 'is_positive', 'n', ' %', ' >'))


@pytest.fixture
def adapter():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(11)
    config = Qwen2Config(vocab_size=261, hidden_size=32, intermediate_size=48,
                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                        max_position_embeddings=2048, attention_dropout=0., pad_token_id=0)
    config._attn_implementation = 'eager'
    return HFAdapter(Qwen2ForCausalLM(config), ByteTokenizer())


def test_predictions_names_prefix(family):
    expected = {'core': (['e','d','d'], ['e','e','d']),
                'swapped': (['e','d','e'], ['d','d','e']),
                'four': (['e','d','d'], ['e','e','d'])}
    assert len(family['views']) == 26
    for config, (sets, counts) in expected.items():
        views = [v for v in family['views'] if v['block'] == 'de' and v['config'] == config and v['naming_variant'] == 0]
        assert [v['expected']['set'] for v in views] == sets
        assert [v['expected']['count'] for v in views] == counts
    for config in ('core', 'four'):
        views = [v for v in family['views'] if v['block'] == 'da' and v['config'] == config and v['naming_variant'] == 0]
        assert [v['donor'] for v in views] == ['C','D']
        assert [v['expected']['set'] for v in views] == ['d','a']
        assert [v['expected']['count'] for v in views] == ['d','d']
        assert len({json.dumps(v['destination'], sort_keys=True) for v in views}) == 1
    for v in family['views']:
        pending = v['names']['pending']
        assert v['source']['code'].endswith(f'{pending} =')
        assert v['destination']['code'].endswith(f"{pending} = {v['prefix']}")
        assert v['src_order'] == SOURCE_ORDERS[v['donor']]
        # completed tasks first, then b: in-order continuation at `=` is b for every donor
        assert v['src_order'][:len(v['src_done'])] == v['src_done']
        assert next(x for x in v['src_order'] if x not in v['src_done']) == 'b'
        assert set(v['names'].values()) == set(NAME_POOL)
        assert 'result =' not in v['source']['code']
        assert 'b' not in v['dst_order']
        for prompt in v['references'].values():
            assert prompt['code'].endswith(f"{pending} = {v['prefix']}")
        assignments = v['source']['code'].splitlines()[2:-1]
        assert [line.strip().split(' =',1)[0] for line in assignments] == [
            v['names'][x] for x in v['src_order'] if x in v['src_done']]
        pred = N.predict(v['src_order'], v['src_done'], v['dst_order'], v['dst_done'])
        assert v['expected'] == {'set': pred['H1'], 'count': pred['H1c']}
        assert set(v['references']) == set(v['readout_roles'])
    for variant in (0, 1):
        views = [v for v in family['views'] if v['naming_variant'] == variant]
        assert len({json.dumps(v['names'], sort_keys=True) for v in views}) == 1
        for donor in ('A','B','C','D'):
            assert len({json.dumps(v['source'], sort_keys=True) for v in views if v['donor']==donor}) == 1
    assert not any(v['block']=='da' and v['config']=='swapped' for v in family['views'])


def test_legacy_h1_remains_source_only_and_d_is_not_relabelled(family):
    assert N.predict('bade','d','ade','a')['H1'] == 'a'
    assert N.predict(SOURCE_ORDERS['D'],'d','ade','a')['H1'] == 'a'   # source order does not change H1
    d = next(v for v in family['views'] if v['donor']=='D')
    assert d['expected']['set']=='a' and d['native']=='d'
    # The a teacher begins before a, rather than silently preserving completed a.
    assert 'n > 0' not in d['references']['a']['code']
    assert 'n > 0' in d['destination']['code']
    assert d['prefix']=='n'  # Never score a using the incompatible n % prefix.


def test_opaque_pending_name_shared_and_varies(family):
    p=N.Pivot('int','is_even','is_divisible_by_three','n % ','2','3')
    q=N.Pivot('int','is_even','is_positive','n',' %',' >')
    seen=set()
    for seed in range(32):
        f=make_family(p,family['role_map'],'train',seed,secondary_pivot=q)
        names=[v['names'] for v in f['views']]
        assert all(n==names[0] for n in names)
        seen.add(names[0]['pending'])
    assert seen==set(NAME_POOL)


def test_block_balancing_and_reciprocal_diagnostic(family):
    from types import SimpleNamespace
    cs=[SimpleNamespace(meta=v) for v in family['views']]
    weights=view_weights(cs)
    for block in ('de','da'):
        assert sum(w for c,w in zip(cs,weights) if c.meta['block']==block)==pytest.approx(.5)
    patterns={(v['expected']['set']!=v['native'],v['expected']['count']!=v['native'])
              for v in family['views'] if v['config']=='core'}
    assert patterns=={(True,True),(False,True),(False,False),(True,False)}


def test_old_or_mixed_design_dataset_rejected():
    data=build_dataset(limit=1)
    for key,value in [('version','set-count-v1'),('renderer','neutral-result-v1'),
                      ('set_rule','destination-union'),('source_order','abde')]:
        bad=copy.deepcopy(data);bad[key]=value;bad.pop('fingerprint');bad['fingerprint']=digest(bad)
        with pytest.raises(ValueError): verify_dataset(bad)
    bad=copy.deepcopy(data);bad['families'][0]['views'].pop()
    bad.pop('fingerprint');bad['fingerprint']=digest(bad)
    with pytest.raises(ValueError,match='incomplete'):verify_dataset(bad)


def test_secondary_pairs_cannot_cross_split():
    data=build_dataset(limit=1)
    other=copy.deepcopy(data['families'][0]);other['split']='dev' if other['split']!='dev' else 'train'
    data['families'].append(other);data.pop('fingerprint');data['fingerprint']=digest(data)
    with pytest.raises(ValueError,match='leakage'): verify_dataset(data)


@pytest.mark.parametrize('prefix,left,right,keep', [
    ('', 'foo', 'bar', False), ('x + ', 'foo', 'bar', False),
    ('s.is', 'lower', 'numeric', True), ('s', '.lower', '.upper', True),
    ("s == '", 'foo', 'bar', True), ('n % ', '2', '3', True),
    ('n ', ' +', '+', False), ('all(x', ' ==', ' %', True),
    ('not', ' any', ' len', True), ('set', '(lst', '(map', True),
    ('all', '(lst', '(x', False)])
def test_contextual_policy(prefix, left, right, keep):
    assert meaningful_contrast(prefix, left, right) == keep


def test_real_pool_deterministic_and_fingerprint(monkeypatch):
    data = build_dataset(limit=5)
    assert verify_dataset(data) is data
    original = N.load_pivots
    monkeypatch.setattr(N, 'load_pivots', lambda *a, **kw: list(reversed(original(*a, **kw))))
    assert build_dataset(limit=5) == data
    assert build_dataset(limit=5, seed=1)['fingerprint'] != data['fingerprint']
    broken = copy.deepcopy(data); broken['families'][0]['prefix'] += ' '
    with pytest.raises(ValueError, match='fingerprint'):
        verify_dataset(broken)
    pivots = original('Qwen/Qwen2.5-Coder-32B-Instruct')
    split = split_pairs(pivots, 0)
    assert split_pairs(list(reversed(pivots)), 0) == split
    for dt in N.BANK:
        assert {v for k,v in split.items() if k.startswith(dt + ':')} == {'train','dev','test'}


def test_chat_preserves_trailing_space(family):
    p = family['views'][0]['destination']
    class StrippingTokenizer(ByteTokenizer):
        def apply_chat_template(self, messages, **kw):
            return super().apply_chat_template(messages, **kw).rstrip()
    assert chat_prompt(StrippingTokenizer(), p).endswith(p['code'])


def test_zero_full_svd_and_partial_grad(adapter):
    layers = [0,1]
    qs = adapter.capture(['a considerably longer source'], layers)
    dst = ['destination']
    clean = adapter.logprobs(dst)
    with patched(adapter, torch.zeros(2,4), layers, qs):
        zero = adapter.logprobs(dst)
    torch.testing.assert_close(clean, zero, rtol=0, atol=0)
    bases = svd_bases(adapter, layers)
    with patched(adapter, torch.zeros(2,4,8), layers, qs, bases):
        svd_zero = adapter.logprobs(dst)
    torch.testing.assert_close(clean, svd_zero, rtol=0, atol=0)
    with patched(adapter, torch.ones(2,4), layers, qs):
        full = adapter.logprobs(dst)
    with patched(adapter, torch.ones(2,4,8), layers, qs, bases):
        svd = adapter.logprobs(dst)
    torch.testing.assert_close(full, svd, rtol=1e-5, atol=1e-6)
    partial = torch.nn.Parameter(torch.full((2,4,8), .4))
    with patched(adapter, partial, layers, qs, bases):
        (-adapter.logprobs(dst)[0,21]).backward()
    assert partial.grad is not None and partial.grad.norm() > 0
    assert all(p.grad is None and not p.requires_grad for p in adapter._model.parameters())
    assert not any(m._forward_hooks for m in adapter._model.modules())


def test_padding(adapter):
    batched = adapter.logprobs(['x', 'longer text'])
    single = adapter.logprobs(['x'])
    torch.testing.assert_close(batched[0], single[0], rtol=1e-5, atol=1e-6)
    inp = adapter.inputs(['x', 'longer text'])
    assert inp['position_ids'][0,-1] == 0 and inp['attention_mask'][0,-1] == 1


def test_absolute_ce_penalizes_other_winner():
    good = torch.tensor([5.,4.,0.]).log_softmax(-1)
    bad = torch.tensor([5.,4.,10.]).log_softmax(-1)
    # Pairwise target/native margin is unchanged; CE catches the third winner.
    torch.testing.assert_close(good[0]-good[1],bad[0]-bad[1])
    assert expected_loss(bad,0,good) > expected_loss(good,0,good)


def test_training_checkpoint_independent_resume(adapter, family, tmp_path):
    # One variant retains both blocks and all thirteen conditions.
    family['views'] = [v for v in family['views'] if v['naming_variant']==0]
    accepted, rejected = validate_and_cache(adapter,[family],[0,1],smoke=True)
    assert not rejected and len(accepted) == 1
    caches = accepted[0][1]
    assert sum(view_weights(caches)) == pytest.approx(1.)
    saved = {}
    for hypothesis in ['set','count']:
        mask = torch.nn.Parameter(torch.ones(2,4))
        opt = torch.optim.AdamW([mask], lr=.03, weight_decay=0.)
        identity = {'hypothesis':hypothesis, 'dataset':'fixture'}
        metrics = train_step(adapter,mask,opt,[0,1],caches,hypothesis)
        assert metrics['gradient_norm'] > 0
        path = tmp_path/f'{hypothesis}.pt'
        save_checkpoint(path, identity, mask, opt,1)
        restored = torch.nn.Parameter(torch.zeros(2,4))
        opt2 = torch.optim.AdamW([restored], lr=.03,weight_decay=0.)
        assert load_checkpoint(path,identity,restored,opt2)[0] == 1
        torch.testing.assert_close(restored,mask,rtol=0,atol=0)
        train_step(adapter,mask,opt,[0,1],caches,hypothesis)
        train_step(adapter,restored,opt2,[0,1],caches,hypothesis)
        torch.testing.assert_close(restored,mask,rtol=0,atol=0)
        with pytest.raises(ValueError,match='identity'):
            load_checkpoint(path,{'hypothesis':'other'},restored,opt2)
        result = evaluate(adapter,mask,[0,1],accepted,hypothesis)
        assert result['n_views'] == 13
        assert {r['block'] for r in result['rows']} == {'de','da'}
        if hypothesis=='set':
            assert all(r['expected']=='a' for r in result['rows'] if r['donor']=='D')
        assert {r['donor'] for r in result['rows']} == {'A','B','C','D'}
        assert all(r['validation']['level'] == 'mechanics_only' for r in result['rows'])
        saved[hypothesis] = mask.detach().clone()
    assert not torch.equal(saved['set'],saved['count'])


def test_validation_rejects_entire_family_on_equal_readouts(family):
    class ConstantAdapter:
        tokenizer = ByteTokenizer()
        def capture(self, prompts, layers):
            return torch.zeros(len(prompts), len(layers), 32)
        def logprobs(self, prompts):
            return torch.zeros(len(prompts),261).log_softmax(-1)
    accepted, rejected = validate_and_cache(ConstantAdapter(),[family],[0,1])
    assert not accepted
    assert rejected[0]['family_id'] == family['id']
    assert len(rejected[0]['reasons']) == len(family['views'])
    assert all('coincide' in r['reason'] for r in rejected[0]['reasons'])


def test_exploratory_training_requires_explicit_opt_in(tmp_path):
    from src.diff.set_count_train import main
    with pytest.raises(ValueError,match='semantic validation'):
        main(['--dataset',str(tmp_path/'does_not_exist.json'),
              '--output-dir',str(tmp_path/'out'),'--hypothesis','set'])


@pytest.mark.parametrize('hypothesis', ['set', 'count'])
def test_control_loss_ablation_weights(family, hypothesis):
    from types import SimpleNamespace
    caches = [SimpleNamespace(meta=v) for v in family['views']]
    base = view_weights(caches)
    supervised = view_weights(caches, hypothesis, 1.)
    primary_only = view_weights(caches, hypothesis, 0.)
    assert supervised == base
    assert any(c.meta['expected'][hypothesis] == c.meta['native'] for c in caches)
    assert any(c.meta['expected'][hypothesis] != c.meta['native'] for c in caches)
    for c, original, ablated in zip(caches, base, primary_only):
        assert ablated == (0. if c.meta['expected'][hypothesis] == c.meta['native'] else original)
    with pytest.raises(ValueError, match='nonnegative'):
        view_weights(caches, hypothesis, -1.)


def test_cli_rejected_resume_preserves_run(adapter, family, tmp_path, monkeypatch):
    from src.diff import set_count_train as T
    from src.diff.set_count_data import VERSION
    monkeypatch.setattr(T, 'load_model', lambda args: adapter)
    family['views'] = [v for v in family['views'] if v['naming_variant']==0]
    dataset = {'version': VERSION, 'renderer': RENDERER, 'source_order': dict(SOURCE_ORDERS),
               'donors': dict(DONORS), 'set_rule': SET_RULE,
               'pair_splits': {k:family['split'] for k in family['pair_groups']}, 'families': [family]}
    dataset['fingerprint'] = digest(dataset)
    path = tmp_path/'data.json'; path.write_text(json.dumps(dataset))
    out = tmp_path/'run'
    args = ['--dataset', str(path), '--output-dir', str(out), '--hypothesis', 'set',
            '--smoke-random', '--max-steps', '0', '--max-families', '1', '--device', 'cpu']
    T.main(args)
    snapshot = {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()}
    bad = args.copy(); bad[bad.index('--hypothesis') + 1] = 'count'
    with pytest.raises(ValueError, match='identity'):
        T.main(bad + ['--resume', str(out/'checkpoint.pt')])
    assert snapshot == {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()}
    with pytest.raises(ValueError, match='fresh directory'):
        T.main(args + ['--validate-only'])
    assert snapshot == {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()}


@pytest.mark.parametrize('value', ['-0.1','1.1','nan'])
def test_mask_initialization_range(tmp_path, value):
    from src.diff.set_count_train import main, parser
    assert parser().parse_args(['--dataset','unused','--output-dir','unused',
                               '--hypothesis','set']).mask_init == 1.
    with pytest.raises(ValueError, match='mask_init'):
        main(['--dataset','unused','--output-dir',str(tmp_path),'--hypothesis','set',
              '--smoke-random','--mask-init',value])
