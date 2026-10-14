"""CPU-only tests: real orchestration/patch kernels with a tiny mocked adapter.

Run: apy -m unittest discover -s set_count_revision_v2/runner -v
No transformers import, download, or large model construction.
"""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PROJECT = Path(os.environ.get('SET_COUNT_TEST_PROJECT', HERE.parents[1] / 'set_count_impl/project')).resolve()
sys.path.insert(0, str(PROJECT))

from smoke_helpers import DATATYPES, aggregate, parse_layers, select_first, signed_changes
import run_real_smoke as runner


class PolicyTests(unittest.TestCase):
    def test_aggregation_distinguishes_control_improvement_primary_regression(self):
        def row(fid, control, loss):
            return {'family_id': fid, 'expected': 'd' if control else 'e', 'native': 'd',
                    'expected_ce': loss, 'expected_correct': control, 'native_correct': True,
                    'target_correct': False, 'expected_logodds': -loss, 'target_native_logodds': -2.}
        initial = [row('a', True, 8.), row('a', False, 2.),
                   row('b', True, 8.), row('b', False, 2.)]
        final = [row('a', True, 2.), row('a', False, 3.),
                 row('b', True, 2.), row('b', False, 3.)]
        a, b = aggregate(initial), aggregate(final)
        self.assertEqual(a['total_expected_ce'], 5.)
        self.assertEqual(b['total_expected_ce'], 2.5)
        self.assertEqual(b['native_expected_control_ce'], 2.)
        self.assertEqual(b['nonnative_primary_ce'], 3.)
        change = signed_changes(a, b)
        self.assertEqual(change['total_expected_ce'], -2.5)
        self.assertEqual(change['nonnative_primary_ce'], 1.)
        self.assertEqual(b['control_ce_contribution'] + b['primary_ce_contribution'], b['total_expected_ce'])
        self.assertEqual(b['whole_family_accuracy'], 0.)

    def test_equal_family_not_variable_batch_weighting(self):
        base = {'expected': 'd', 'native': 'd', 'expected_ce': 10., 'expected_correct': True,
                'native_correct': True, 'target_correct': False, 'expected_logodds': 1., 'target_native_logodds': -1.}
        rows = [dict(base, family_id='a')]
        rows += [dict(base, family_id='b', expected_ce=2.) for _ in range(3)]
        result = aggregate(rows)
        self.assertAlmostEqual(result['total_expected_ce'], 6.)
        self.assertIsNone(result['nonnative_primary_ce'])
        with self.assertRaises(ValueError):
            aggregate([])

    def test_first_passing_per_datatype_no_donor_cherry_picking(self):
        candidates = {dt: [{'id': f'{dt}-{i}'} for i in range(4)] for dt in DATATYPES}
        calls = []
        def validate(fs):
            f, = fs
            calls.append(f['id'])
            if f['id'].endswith('-1'):
                return [(f, list(range(13)))], []
            return [], [{'family_id': f['id'], 'reason': 'one failed view rejects family'}]
        selected, rejected, attempts = select_first(candidates, validate)
        self.assertEqual(calls, [f'{dt}-{i}' for dt in DATATYPES for i in range(2)])
        self.assertEqual([f['id'] for f, _ in selected], [f'{dt}-1' for dt in DATATYPES])
        self.assertTrue(all(len(views) == 13 for _, views in selected))
        self.assertEqual(len(rejected), 3)
        selected2, _, attempts2 = select_first(candidates, validate)
        self.assertEqual(selected, selected2)
        self.assertEqual(attempts, attempts2)

    def test_all_fail_does_not_manufacture_a_target(self):
        candidates = {dt: [{'id': dt}] for dt in DATATYPES}
        selected, rejected, _ = select_first(candidates,
            lambda fs: ([], [{'family_id': fs[0]['id'], 'reason': 'same real token'}]))
        self.assertEqual(selected, [])
        self.assertEqual(len(rejected), 3)

    def test_hierarchical_weights_not_flat_block_counts(self):
        base = {'family_id': 'a', 'expected': 'd', 'native': 'd',
                'expected_correct': True, 'native_correct': True, 'target_correct': False,
                'expected_logodds': 1., 'target_native_logodds': -1.}
        rows = [dict(base, block='de', expected_ce=10., view_weight=1/18) for _ in range(9)]
        rows += [dict(base, block='da', expected_ce=2., view_weight=1/8) for _ in range(4)]
        self.assertAlmostEqual(aggregate(rows)['total_expected_ce'], 6.)
        self.assertNotAlmostEqual(aggregate(rows)['total_expected_ce'], (9*10 + 4*2)/13)
        # Row weights are normalized per family before equal-family averaging.
        rows += [dict(base, family_id='b', expected_ce=4., view_weight=7.)]
        self.assertAlmostEqual(aggregate(rows)['total_expected_ce'], 5.)
        rows[0].pop('view_weight')
        with self.assertRaisesRegex(ValueError, 'mixed'):
            aggregate(rows)

    def test_reject_invalid_row_weights(self):
        for weight in (-1., 0., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                aggregate([{'family_id': 'a', 'view_weight': weight}])

    def test_diagnostics_use_each_views_readout_tokens(self):
        adapter = SimpleNamespace(tokenizer=None, readout_audit={
            'd ref': {'top_string': 'D'}, 'a ref': {'top_string': 'A'}})
        trainer = SimpleNamespace(chat_prompt=lambda tokenizer, text: text)
        family = {'pool_tokens': {'d': 'D', 'e': 'E'}, 'views': [{
            'id': 'da', 'block': 'da', 'pool_tokens': {'d': 'D', 'a': 'A'},
            'references': {'d': 'd ref', 'a': 'a ref'}, 'source': 'src', 'destination': 'dst'}]}
        row, = runner.readout_diagnostics(family, adapter, trainer)
        self.assertEqual(row['stored_vs_real'], {'d': True, 'a': True})
        self.assertEqual(row['stored_tokens'], {'d': 'D', 'a': 'A'})

    def test_candidates_preserve_train_splits_and_prefer_known_pair(self):
        views = [{'block': block, 'config': config, 'donor': donor}
                 for block, configs, donors in (
                     ('de', ('core', 'swapped', 'four'), ('A', 'B', 'C')),
                     ('da', ('core', 'four'), ('C', 'D')))
                 for config in configs for donor in donors]
        def family(fid, split='train', preferred=False):
            return {'id': fid, 'datatype': 'str', 'split': split, 'views': copy.deepcopy(views),
                    'role_map': {'a': 'a', 'b': 'b', 'd': 'wanted' if preferred else 'd', 'e': 'e', 'f': 'f'},
                    'prefix': 'prefix', 'pool_tokens': {'d': 'D', 'e': 'E'}}
        families = [family(f'other{i}') for i in range(6)]
        families += [family('preferred', preferred=True), family('heldout', 'test', True)]
        dataset = {'version': 'set-count-v2', 'renderer': 'opaque-identifiers-v2', 'families': families}
        calls = []
        def build(**kwargs):
            calls.append((Path.cwd(), kwargs))
            return copy.deepcopy(dataset)
        data = SimpleNamespace(build_dataset=build, digest=lambda value: json.dumps(value))
        with tempfile.TemporaryDirectory() as tmp:
            examples = Path(tmp) / 'examples.json'
            examples.write_text(json.dumps({'examples': [{
                'datatype': 'str', 'roles': families[-2]['role_map'], 'prefix': 'prefix',
                'pool_record': {'dest_top_token': 'D', 'target_top_token': 'E'}}]}))
            with self.assertLogs(runner.LOG, level='WARNING') as logs:
                candidates = runner.build_candidates(tmp, examples, data, 7)
            self.assertEqual(calls, [(Path(tmp), dict(model=runner.MODEL, seed=7,
                                                      n_per_pivot=1, naming_variants=1))])
            self.assertEqual(len(candidates['str']), 4)
            self.assertEqual(candidates['str'][0]['id'], 'preferred')
            self.assertTrue(all(f['split'] == 'train' for f in candidates['str']))
            self.assertEqual(candidates['int'], [])
            self.assertEqual(candidates['list[int]'], [])
            self.assertTrue(any('int' in line for line in logs.output))
            self.assertEqual(dataset['families'], families)  # no split mutation in source data
            dataset['families'] = [family('broken')]
            dataset['families'][0]['views'].pop()
            with self.assertRaisesRegex(ValueError, 'complete two-block'):
                runner.build_candidates(tmp, None, data, 7)
            dataset['version'] = 'set-count-v1'
            with self.assertRaisesRegex(ValueError, 'set-count-v2'):
                runner.build_candidates(tmp, None, data, 7)

    def test_cli_layers(self):
        self.assertEqual(parse_layers('20..55'), list(range(20, 56)))
        self.assertEqual(parse_layers('0,3,63'), [0, 3, 63])
        for invalid in ('55..20', '0,0', '-1', '64'):
            with self.assertRaises(ValueError):
                parse_layers(invalid)
        args = runner.parser().parse_args(['--model-dir', 'weights', '--output-dir', 'out', '--project-root', '.'])
        self.assertEqual(args.steps, 12)
        self.assertIsNone(args.examples_json)


class ResumeProvenanceTests(unittest.TestCase):
    """Trusted tiny CPU checkpoint payloads; never load a pretrained model."""
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(1)

    def prior_output(self, out):
        out.mkdir()
        identity = {'source_hashes': {'runner': 'v2', 'trainer': 'v2'},
                    'pool_sha256': 'pool-v2', 'selected_fingerprint': 'two-block-families',
                    'readout_fingerprint': 'binary-readouts'}
        # Include all overwritten preflight evidence, plus telemetry and failures.
        names = ('protocol.json', 'protocol_resume_attempt.json', 'candidate_families.json',
                 'selection_attempts.json', 'selected_families.json', 'selected_prompts.json',
                 'rejected_families.json', 'validation_readouts.json', 'validation_cache.pt',
                 'clean_baselines.json', 'representative_continuations.json',
                 'summary.json', 'wandb.json', 'failure.json')
        for name in names:
            (out / name).write_bytes(b'original evidence: ' + name.encode())
        for hypothesis, kind in runner.CASES:
            case_dir = out / f'{hypothesis}_{kind}'
            case_dir.mkdir()
            runner.write_json(case_dir / 'progress.json', {
                'training': [{'step': 1, 'loss': 2.}], 'evaluations': [{'step': 0}],
                'complete': True})
            for name in ('summary.json', 'zero_mask_checks.json', 'failure.json'):
                (case_dir / name).write_bytes(b'original case evidence')
            for name, step in (('initial_checkpoint.pt', 0), ('checkpoint.pt', 1)):
                self.torch.save({'schema': 1, 'identity': {
                    **identity, 'hypothesis': hypothesis, 'mask_kind': kind},
                    'step': step, 'mask': self.torch.ones(1),
                    'bases': {0: {'q': self.torch.ones(1)}} if kind == 'svd' else None},
                    case_dir / name)
        return identity, names

    def snapshot(self, out):
        return {p.relative_to(out).as_posix(): p.read_bytes()
                for p in out.rglob('*') if p.is_file()}

    def test_all_case_identities_and_step_ceilings_preserve_old_bytes(self):
        # A mismatch in the LAST case must block publication and optimization
        # just as one in the first case does. Check initial checkpoints as well.
        for case in runner.CASES:
            for checkpoint in ('initial_checkpoint.pt', 'checkpoint.pt'):
                for mismatch in ('v1', 'source', 'data', 'readout', 'step'):
                    with self.subTest(case=case, checkpoint=checkpoint, mismatch=mismatch):
                        with tempfile.TemporaryDirectory() as tmp:
                            out, attempt = Path(tmp) / 'old', Path(tmp) / 'attempt'
                            identity, names = self.prior_output(out)
                            path = out / '_'.join(case) / checkpoint
                            state = self.torch.load(path, weights_only=False)
                            if mismatch == 'v1':
                                state['identity']['source_hashes'] = {'runner': 'v1', 'trainer': 'v1'}
                                state['identity']['selected_fingerprint'] = 'one-block-families'
                            elif mismatch == 'source':
                                state['identity']['source_hashes']['trainer'] = 'changed-source'
                            elif mismatch == 'data':
                                state['identity']['pool_sha256'] = 'different-pool'
                            elif mismatch == 'readout':
                                state['identity']['readout_fingerprint'] = 'different-tokens'
                            else:
                                state['step'] = 4
                            self.torch.save(state, path)
                            before = self.snapshot(out)
                            attempt.mkdir()
                            for name in names:
                                (attempt / name).write_bytes(b'new preflight evidence')
                            with self.assertRaisesRegex(ValueError, 'identity mismatch|total steps'):
                                runner.publish_resume_preflight(attempt, out, identity, 3, self.torch)
                            self.assertEqual(self.snapshot(out), before)
                            self.assertEqual(len(list(attempt.iterdir())), len(names))

    def test_resume_without_checkpoint_does_not_replace_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, attempt = Path(tmp) / 'old', Path(tmp) / 'attempt'
            out.mkdir(); attempt.mkdir()
            (out / 'protocol.json').write_bytes(b'old v1 evidence without weights')
            (attempt / 'protocol.json').write_bytes(b'new v2 evidence')
            before = self.snapshot(out)
            with self.assertRaisesRegex(ValueError, 'no checkpoint'):
                runner.publish_resume_preflight(attempt, out, {}, 3, self.torch)
            self.assertEqual(self.snapshot(out), before)

    def test_compatible_publication_preserves_metrics_and_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, attempt = Path(tmp) / 'old', Path(tmp) / 'attempt'
            identity, _ = self.prior_output(out)
            before = self.snapshot(out)
            attempt.mkdir()
            for name in ('protocol.json', 'validation_cache.pt'):
                (attempt / name).write_bytes(b'compatible fresh preflight')
            runner.publish_resume_preflight(attempt, out, identity, 3, self.torch)
            after = self.snapshot(out)
            for name, content in before.items():
                self.assertEqual(after[name], b'compatible fresh preflight'
                                 if name in ('protocol.json', 'validation_cache.pt') else content)
            self.assertEqual(list(attempt.iterdir()), [])

    def test_missing_resume_history_rejects_before_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, attempt = Path(tmp) / 'old', Path(tmp) / 'attempt'
            identity, _ = self.prior_output(out)
            (out / 'count_svd' / 'progress.json').unlink()
            before = self.snapshot(out)
            attempt.mkdir()
            (attempt / 'protocol.json').write_bytes(b'new')
            with self.assertRaisesRegex(ValueError, 'history metrics'):
                runner.publish_resume_preflight(attempt, out, identity, 3, self.torch)
            self.assertEqual(self.snapshot(out), before)

    def test_repeated_rejected_main_attempts_keep_prior_failure_and_wandb(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'old'
            identity, names = self.prior_output(out)
            before = self.snapshot(out)
            attempts = []
            def preflight(args, work_out, resume_out=None):
                self.assertEqual(resume_out, out)
                self.assertNotEqual(work_out, out)
                attempts.append(work_out)
                for name in names:
                    # The actual resume preflight does not create a W&B record.
                    if name not in ('wandb.json', 'failure.json'):
                        (work_out / name).write_bytes(b'new attempt evidence')
                runner.publish_resume_preflight(work_out, resume_out,
                    dict(identity, pool_sha256='mismatch'), args.steps, self.torch)
                runner.WandbLogger(args.experiment_id, resume_out, {}, True)
                runner.run_case()  # must never be reached
            with patch.object(runner, 'execute_attempt', side_effect=preflight), \
                    patch.object(runner, 'WandbLogger') as logger, \
                    patch.object(runner, 'run_case') as optimize, \
                    self.assertLogs(runner.LOG, level='ERROR'):
                for _ in range(2):
                    with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                        runner.main(['--model-dir', tmp, '--project-root', str(PROJECT),
                                     '--output-dir', str(out), '--resume', '--experiment-id', 'test'])
                    self.assertEqual(self.snapshot(out), before)
                logger.assert_not_called()
                optimize.assert_not_called()
            self.assertEqual(len(set(attempts)), 2)
            for attempt in attempts:
                failure = json.loads((attempt / 'failure.json').read_text())
                self.assertIn('identity mismatch', failure['traceback'])
                self.assertFalse((attempt / 'wandb.json').exists())

    def test_actual_execute_preflight_rejection_never_opens_wandb_or_optimizes(self):
        from contextlib import ExitStack
        from src.diff import set_count_train as trainer
        torch = self.torch
        parameter = torch.ones(1, dtype=torch.bfloat16)
        model = SimpleNamespace(training=False, parameters=lambda: iter([parameter]),
            named_modules=lambda: iter([]), config=SimpleNamespace(
                _attn_implementation='eager', to_dict=lambda: {}))
        model.to = lambda device: model
        tokenizer = SimpleNamespace(init_kwargs={})
        logits = torch.zeros(1, 3)
        adapter = SimpleNamespace(_model=model, tokenizer=tokenizer, n_layer=64,
            n_heads=40, n_kv_heads=8, head_dim=128, readout_audit={},
            logprobs=lambda prompts: logits)
        family = {'id': 'new-v2-family', 'datatype': 'str', 'views': [{'destination': 'dst'}]}
        cache = SimpleNamespace(meta={'id': 'view', 'source': 'src', 'references': {}},
            destination='dst', token_ids={'d': 0, 'a': 1}, validation={})
        transformers = SimpleNamespace(__version__='mock',
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *a, **kw: model))
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            out = Path(tmp) / 'old'
            self.prior_output(out)
            before = self.snapshot(out)
            stack.enter_context(patch.dict(sys.modules, {'transformers': transformers}))
            stack.enter_context(patch.object(torch.cuda, 'is_available', return_value=True))
            stack.enter_context(patch.object(torch.cuda, 'device_count', return_value=1))
            stack.enter_context(patch.object(torch.cuda, 'manual_seed_all'))
            stack.enter_context(patch.object(torch.cuda, 'get_device_name', return_value='mock'))
            stack.enter_context(patch.object(torch.cuda, 'get_device_properties',
                                            return_value=SimpleNamespace(total_memory=1)))
            for name in ('reset_peak', 'synchronize'):
                stack.enter_context(patch.object(runner, name))
            stack.enter_context(patch.object(runner, 'memory', return_value={}))
            stack.enter_context(patch.object(runner.importlib.metadata, 'version', return_value='mock'))
            stack.enter_context(patch.object(runner, 'build_candidates',
                return_value={dt: [family] if dt == 'str' else [] for dt in DATATYPES}))
            stack.enter_context(patch.object(runner, 'make_adapter_class',
                                            return_value=lambda *a, **kw: adapter))
            stack.enter_context(patch.object(trainer.HFAdapter, 'logprobs', return_value=logits))
            stack.enter_context(patch.object(trainer, 'chat_prompt', side_effect=lambda tok, p: p))
            stack.enter_context(patch.object(trainer, 'validate_and_cache',
                                            return_value=([(family, [cache])], [])))
            stack.enter_context(patch.object(runner, 'readout_diagnostics', return_value=[]))
            stack.enter_context(patch.object(runner, 'cached_clean_baseline', return_value={}))
            generation = stack.enter_context(patch.object(runner, 'generate_examples',
                side_effect=lambda *a: runner.write_json(a[-1] / 'representative_continuations.json', [])))
            logger = stack.enter_context(patch.object(runner, 'WandbLogger'))
            optimize = stack.enter_context(patch.object(runner, 'run_case'))
            stack.enter_context(self.assertLogs(runner.LOG, level='ERROR'))
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                runner.main(['--model-dir', tmp, '--project-root', str(PROJECT),
                             '--output-dir', str(out), '--resume', '--experiment-id', 'test'])
            generation.assert_called_once()  # actual execute reached staged preflight's end
            logger.assert_not_called()
            optimize.assert_not_called()
            self.assertEqual(self.snapshot(out), before)
            attempt, = Path(tmp).glob('.old-resume-*')
            self.assertTrue((attempt / 'validation_cache.pt').exists())
            self.assertTrue((attempt / 'protocol.json').exists())
            self.assertTrue((attempt / 'failure.json').exists())

    def test_fresh_execute_path_is_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            args = SimpleNamespace(resume=False)
            with patch.object(runner, 'execute_attempt', return_value='fresh') as execute:
                self.assertEqual(runner.execute(args, out), 'fresh')
                execute.assert_called_once_with(args, out)
            self.assertEqual(list(out.iterdir()), [])


class MockAdapterIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from src.diff import set_count_train as trainer, set_count_data as data
        cls.torch, cls.trainer, cls.data = torch, trainer, data
        torch.set_num_threads(1)

    def fixture(self):
        torch, trainer = self.torch, self.trainer
        torch.manual_seed(12)
        class TinyModel(torch.nn.Module):
            """Mock forward graph with Q/K module names, not a pretrained model."""
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(261, 4)
                self.model = torch.nn.Module()
                self.model.layers = torch.nn.ModuleList()
                for _ in range(2):
                    layer = torch.nn.Module()
                    layer.self_attn = torch.nn.Module()
                    layer.self_attn.q_proj = torch.nn.Linear(4, 4)
                    layer.self_attn.k_proj = torch.nn.Linear(4, 2)
                    self.model.layers.append(layer)
                self.lm_head = torch.nn.Linear(4, 261)
                self.config = SimpleNamespace(model_type='qwen2', num_attention_heads=2,
                    num_key_value_heads=1, num_hidden_layers=2, hidden_size=4)
            def forward(self, input_ids, logits_to_keep=0, **kwargs):
                h = self.embedding(input_ids)
                for layer in self.model.layers:
                    h = (h + layer.self_attn.q_proj(h)).tanh()
                if logits_to_keep:
                    h = h[:, -logits_to_keep:]
                return SimpleNamespace(logits=self.lm_head(h))
        adapter = runner.make_adapter_class(trainer, torch)(TinyModel(), trainer.ByteTokenizer())
        adapter.audit_enabled = False
        caches = []
        with torch.no_grad():
            clean = adapter.logprobs(['destination'])[0].cpu()
            q = adapter.capture(['source'], [0, 1])
        for block, configs, donors, other in (
                ('de', ('core', 'swapped', 'four'), ('A', 'B', 'C'), 'e'),
                ('da', ('core', 'four'), ('C', 'D'), 'a')):
            for config in configs:
                for donor in donors:
                    meta = {'id': f'toy/{block}/{config}/{donor}', 'block': block,
                            'config': config, 'donor': donor, 'naming_variant': 0,
                            'source': 'source', 'destination': 'destination',
                            'readout_roles': ['d', other], 'native': 'd',
                            'source_order': 'bade', 'source_current': donor,
                            'destination_order': config, 'destination_current': 'a',
                            'expected': {'set': other if donor in ('A', 'D') else 'd',
                                         'count': other if donor in ('A', 'B') else 'd'}}
                    caches.append(trainer.ViewCache(meta, 'destination', q,
                        {'d': 3, other: 4}, {'d': clean, other: clean}, clean, {'level': 'mock-only'}))
        family = {'id': 'toy', 'split': 'train', 'datatype': 'str'}
        return adapter, [(family, caches)]

    def logger(self):
        class Log:
            def __init__(self):
                self.records = []
            def log(self, *args):
                self.records.append(args)
        return Log()

    def test_real_pool_candidates_are_deterministic_bounded_complete(self):
        examples = None
        a = runner.build_candidates(PROJECT, examples, self.data, 0)
        b = runner.build_candidates(PROJECT, examples, self.data, 0)
        self.assertEqual(a, b)
        self.assertEqual(tuple(a), DATATYPES)
        for dt in DATATYPES:
            self.assertLessEqual(len(a[dt]), 4)
            self.assertGreater(len(a[dt]), 0)
            for f in a[dt]:
                self.assertEqual(f['split'], 'train')
                self.assertEqual(len(f['views']), 13)
                self.assertFalse(any('token_id' in v for v in f['views']))
                self.assertEqual({v['block'] for v in f['views']}, {'de', 'da'})
                self.assertEqual(sum(v['block'] == 'de' for v in f['views']), 9)
                self.assertEqual(sum(v['block'] == 'da' for v in f['views']), 4)

    def test_cached_clean_baseline_equals_zero_head_mask_evaluation(self):
        adapter, accepted = self.fixture()
        for hypothesis in ('set', 'count'):
            cached = runner.cached_clean_baseline(accepted, hypothesis, self.trainer)
            actual = runner.evaluate_fixed(adapter, self.torch.zeros(2, 2), [0, 1],
                accepted, hypothesis, None, self.trainer)
            self.assertEqual(cached['summary'], actual['summary'])
            for cached_row, actual_row in zip(cached['rows'], actual['rows']):
                for key in actual_row.keys() & cached_row.keys():
                    self.assertEqual(cached_row[key], actual_row[key], key)
            self.assertEqual(len(cached['rows']), 13)
            da_rows = [r for r in cached['rows'] if r['block'] == 'da']
            self.assertTrue(all(r['target'] == 'a' for r in da_rows))
            self.assertEqual({r['expected'] for r in da_rows}, {'a', 'd'} if hypothesis == 'set' else {'d'})
            for block in ('de', 'da'):
                self.assertAlmostEqual(sum(r['view_weight'] for r in cached['rows'] if r['block'] == block), .5)

    def test_cached_baseline_with_a_native_and_d_target(self):
        adapter, accepted = self.fixture()
        for c in accepted[0][1]:
            if c.meta['block'] == 'da':
                c.meta['native'] = 'a'
        cached = runner.cached_clean_baseline(accepted, 'set', self.trainer)
        actual = runner.evaluate_fixed(adapter, self.torch.zeros(2, 2), [0, 1],
            accepted, 'set', None, self.trainer)
        self.assertEqual(cached['summary'], actual['summary'])
        self.assertTrue(all(r['target'] == 'd' for r in cached['rows'] if r['block'] == 'da'))

    def test_last_token_mock_equivalence(self):
        adapter, _ = self.fixture()
        with self.torch.no_grad():
            ordinary = self.trainer.HFAdapter.logprobs(adapter, ['hello'])
            optimized = adapter.logprobs(['hello'])
        self.torch.testing.assert_close(ordinary, optimized)

    def test_four_cases_fixed_eval_checkpoints_and_no_base_gradients(self):
        from src.diff.query_mask_ops import svd_bases
        adapter, accepted = self.fixture()
        snapshot = runner.hook_snapshot(adapter._model)
        original = {k: v.clone() for k, v in adapter._model.state_dict().items()}
        bases = svd_bases(adapter, [0, 1])
        with tempfile.TemporaryDirectory() as tmp:
            for hypothesis, kind in runner.CASES:
                case_dir = Path(tmp) / f'{hypothesis}_{kind}'
                result = runner.run_case(case_dir, hypothesis, kind, adapter, accepted,
                    [0, 1], bases if kind == 'svd' else None, self.trainer, self.torch,
                    self.logger(), {'fixture': 'mock'}, steps=4)
                self.assertEqual([x['step'] for x in result['evaluations']], [0, 3, 4])
                self.assertEqual(len(result['training']), 4)
                self.assertTrue(result['complete'])
                self.assertEqual(result['initial']['summary']['n_views'], 13)
                self.assertEqual(result['final']['summary']['n_views'], 13)
                self.assertEqual(result['training'][0]['family_id'], 'toy')
                saved = self.torch.load(case_dir / 'checkpoint.pt', weights_only=False)
                initial = self.torch.load(case_dir / 'initial_checkpoint.pt', weights_only=False)
                self.assertEqual(saved['step'], 4)
                self.assertEqual(initial['step'], 0)
                self.assertTrue(bool((initial['mask'] == 1).all()))
                self.assertEqual(saved['identity']['hypothesis'], hypothesis)
                runner.assert_frozen(adapter._model)
                runner.assert_hooks_clean(adapter._model, snapshot)
        for key, value in adapter._model.state_dict().items():
            self.torch.testing.assert_close(value, original[key], rtol=0, atol=0)

    def test_resume_matches_uninterrupted(self):
        adapter, accepted = self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            resume_dir, full_dir = Path(tmp) / 'resume', Path(tmp) / 'full'
            args = ('set', 'head', adapter, accepted, [0, 1], None, self.trainer,
                    self.torch, self.logger(), {'fixture': 'resume'})
            runner.run_case(resume_dir, *args, steps=3)
            resumed = runner.run_case(resume_dir, *args, steps=6, resume=True)
            full = runner.run_case(full_dir, *args, steps=6)
            self.assertEqual([x['step'] for x in resumed['evaluations']], [0, 3, 6])
            self.assertEqual(resumed['final'], full['final'])
            a = self.torch.load(resume_dir / 'checkpoint.pt', weights_only=False)
            b = self.torch.load(full_dir / 'checkpoint.pt', weights_only=False)
            self.torch.testing.assert_close(a['mask'], b['mask'], rtol=0, atol=0)

    def test_failure_keeps_initial_checkpoint_and_traceback(self):
        adapter, accepted = self.fixture()
        baseline = runner.hook_snapshot(adapter._model)
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp)
            with patch.object(self.trainer, 'train_step', side_effect=RuntimeError('mock injected failure')):
                with self.assertRaisesRegex(RuntimeError, 'mock injected failure'):
                    runner.run_case(case_dir, 'count', 'head', adapter, accepted, [0, 1], None,
                        self.trainer, self.torch, self.logger(), {'fixture': 'failure'}, steps=3)
            failure = json.loads((case_dir / 'failure.json').read_text())
            self.assertIn('mock injected failure', failure['traceback'])
            self.assertTrue((case_dir / 'initial_checkpoint.pt').exists())
            self.assertEqual(self.torch.load(case_dir / 'checkpoint.pt', weights_only=False)['step'], 0)
            runner.assert_hooks_clean(adapter._model, baseline)

    def test_no_descent_is_not_an_exit_failure(self):
        adapter, accepted = self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            result = runner.run_case(Path(tmp), 'set', 'head', adapter, accepted, [0, 1], None,
                self.trainer, self.torch, self.logger(), {'fixture': 'no-descent'}, steps=0)
            self.assertTrue(result['complete'])
            self.assertFalse(result['loss_descent_observed'])
            self.assertEqual(result['signed_change_final_minus_initial']['total_expected_ce'], 0.)


if __name__ == '__main__':
    unittest.main()
