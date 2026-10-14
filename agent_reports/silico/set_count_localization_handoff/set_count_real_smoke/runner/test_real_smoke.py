"""CPU-only tests: real orchestration/patch kernels with a tiny mocked adapter.

Run: apy -m unittest discover -s set_count_real_smoke/runner -v
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
                return [(f, list(range(9)))], []
            return [], [{'family_id': f['id'], 'reason': 'one failed view rejects family'}]
        selected, rejected, attempts = select_first(candidates, validate)
        self.assertEqual(calls, [f'{dt}-{i}' for dt in DATATYPES for i in range(2)])
        self.assertEqual([f['id'] for f, _ in selected], [f'{dt}-1' for dt in DATATYPES])
        self.assertTrue(all(len(views) == 9 for _, views in selected))
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

    def test_cli_layers(self):
        self.assertEqual(parse_layers('20..55'), list(range(20, 56)))
        self.assertEqual(parse_layers('0,3,63'), [0, 3, 63])
        for invalid in ('55..20', '0,0', '-1', '64'):
            with self.assertRaises(ValueError):
                parse_layers(invalid)
        args = runner.parser().parse_args(['--model-dir', 'weights', '--output-dir', 'out', '--project-root', '.'])
        self.assertEqual(args.steps, 12)


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
        for config in ('core', 'swapped', 'four'):
            for donor in ('A', 'B', 'C'):
                meta = {'id': f'toy/{config}/{donor}', 'config': config, 'donor': donor,
                        'native': 'd', 'expected': {'set': 'e' if donor == 'A' else 'd',
                                                  'count': 'e' if donor != 'C' else 'd'}}
                caches.append(trainer.ViewCache(meta, 'destination', q,
                    {'d': 3, 'e': 4}, {'d': clean, 'e': clean}, clean, {'level': 'mock-only'}))
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
        examples = HERE.parents[1] / 'set_count_design/concrete_examples.json'
        a = runner.build_candidates(PROJECT, examples, self.data, 0)
        b = runner.build_candidates(PROJECT, examples, self.data, 0)
        self.assertEqual(a, b)
        self.assertEqual(tuple(a), DATATYPES)
        for dt in DATATYPES:
            self.assertEqual(len(a[dt]), 4)
            for f in a[dt]:
                self.assertEqual(f['split'], 'train')
                self.assertEqual(len(f['views']), 9)
                self.assertTrue(f['prefix'].endswith(' ') if dt == 'int' else True)
                self.assertFalse(any('token_id' in v for v in f['views']))

    def test_cached_clean_baseline_equals_zero_head_mask_evaluation(self):
        adapter, accepted = self.fixture()
        for hypothesis in ('set', 'count'):
            cached = runner.cached_clean_baseline(accepted, hypothesis)
            actual = runner.evaluate_fixed(adapter, self.torch.zeros(2, 2), [0, 1],
                accepted, hypothesis, None, self.trainer)
            self.assertEqual(cached['summary'], actual['summary'])
            self.assertEqual(len(cached['rows']), 9)

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
                self.assertEqual(result['initial']['summary']['n_views'], 9)
                self.assertEqual(result['final']['summary']['n_views'], 9)
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
