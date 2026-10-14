#!/usr/bin/env python3
"""Actual frozen Qwen2.5-Coder-32B bounded training-only calibration smoke.

No downloads, artificial labels, semantic correctness, or generalization claim.
Loss decrease is measured, never required for successful execution. All four
cases use the same preselected families and the same clean/Q caches.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import itertools
import json
import logging
import math
import os
from pathlib import Path
import platform
import random
import sys
import time
import traceback

from smoke_helpers import (CASES, DATATYPES, MODEL, REVISION, aggregate,
                           parse_layers, select_first, signed_changes)

LOG = logging.getLogger('real32b-smoke')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(temporary, path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def working_directory(path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def build_candidates(project_root, examples_json, data, seed):
    """One known assignment + three fixed auxiliary/name assignments per datatype."""
    with working_directory(project_root):
        pivots = data.N.load_pivots(MODEL, core_only=True)
    examples = json.loads(Path(examples_json).read_text())['examples']
    by_type = {e['datatype']: e for e in examples}
    candidates = {}
    for datatype in DATATYPES:
        example = by_type[datatype]
        roles = example['roles']
        matches = [p for p in pivots if p.datatype == datatype
                   and p.b == roles['d'] and p.d == roles['e']
                   and p.prefix == example['prefix']]
        if not matches:
            raise ValueError(f'known {datatype} pair/prefix is absent from the actual pool')
        # Disambiguate using the actual stored known tokens; never invent token IDs.
        record = example['pool_record']
        matches = [p for p in matches if p.b_tok == record['dest_top_token']
                   and p.d_tok == record['target_top_token']]
        if not matches:
            raise ValueError(f'known {datatype} pool-token provenance differs')
        pivot = sorted(set(matches), key=lambda p: data.digest(p.__dict__))[0]
        assignments = [roles.copy()]
        other = sorted(x for x in data.N.BANK[datatype] if x not in (pivot.b, pivot.d))
        alternatives = []
        for a, b, f in itertools.permutations(other, 3):
            if (data.N._fam(a) in {data.N._fam(pivot.b), data.N._fam(pivot.d)} or
                    data.N._fam(b) in {data.N._fam(a), data.N._fam(pivot.b), data.N._fam(pivot.d)}):
                continue
            r = dict(a=a, b=b, d=pivot.b, e=pivot.d, f=f)
            if r != roles:
                alternatives.append(r)
        alternatives.sort(key=lambda r: data.digest([seed, datatype, r]))
        assignments.extend(alternatives[:3])
        candidates[datatype] = []
        for index, assignment in enumerate(assignments):
            f = data.make_family(pivot, assignment, 'train', seed + index, naming_variants=1)
            f['calibration_provenance'] = {'candidate_index': index, 'known_assignment': index == 0,
                'designation': 'train-only calibration; previously explored source bank',
                'original_pair_split_not_used': True, 'semantic_correctness_unverified': True}
            assert len(f['views']) == 9
            assert {(v['config'], v['donor']) for v in f['views']} == set(itertools.product(
                ('core', 'swapped', 'four'), ('A', 'B', 'C')))
            candidates[datatype].append(f)
    return candidates


def make_adapter_class(trainer, torch):
    class LastTokenAdapter(trainer.HFAdapter):
        """Same adapter behavior, avoiding full sequence x vocabulary logits."""
        audit_enabled = True

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.readout_audit = {}

        def logprobs(self, prompts):
            lp = self._model(**self.inputs(prompts), logits_to_keep=1).logits[:, -1].float().log_softmax(-1)
            if self.audit_enabled:
                for prompt, values in zip(prompts, lp.detach()):
                    token_id = int(values.argmax())
                    self.readout_audit[prompt] = {'top_id': token_id,
                        'top_string': self.tokenizer.decode([token_id], clean_up_tokenization_spaces=False)}
            return lp
    return LastTokenAdapter


def hook_snapshot(model):
    return {name: set(module._forward_hooks) for name, module in model.named_modules()}


def assert_hooks_clean(model, baseline):
    if hook_snapshot(model) != baseline:
        raise RuntimeError('patch hook leaked across evaluation/training/generation boundary')


def remove_new_hooks(model, baseline):
    """Exception cleanup removes only hooks added by this harness's patch calls."""
    for name, module in model.named_modules():
        for key in set(module._forward_hooks) - baseline.get(name, set()):
            del module._forward_hooks[key]


def assert_frozen(model):
    if model.training or any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise RuntimeError('base model must remain frozen, in eval mode, without gradients')


def synchronize(torch):
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def memory(torch):
    if not torch.cuda.is_available():
        return {'cuda_peak_allocated_bytes': None, 'cuda_peak_reserved_bytes': None}
    return {'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved()}


def reset_peak(torch):
    synchronize(torch)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def readout_diagnostics(family, adapter, trainer):
    rows = []
    for v in family['views']:
        refs = {k: adapter.readout_audit.get(trainer.chat_prompt(adapter.tokenizer, p))
                for k, p in v['references'].items()}
        rows.append({'view_id': v['id'], 'stored_tokens': family['pool_tokens'],
                     'real_reference_readouts': refs,
                     'stored_vs_real': {k: value['top_string'] == family['pool_tokens'][k]
                                        if value else None for k, value in refs.items()},
                     'destination': adapter.readout_audit.get(trainer.chat_prompt(adapter.tokenizer, v['destination'])),
                     'source': adapter.readout_audit.get(trainer.chat_prompt(adapter.tokenizer, v['source'])),
                     'semantic_correctness_unverified': True})
    return rows


def generate_examples(adapter, accepted, trainer, torch, baseline, out):
    """Small fixed deterministic descriptive continuations, never used to select."""
    records = []
    for family, _ in accepted:
        # One representative donor/source + recipient and both references.
        view = family['views'][0]
        prompts = [('source_A', view['source']), ('recipient', view['destination'])]
        prompts += [(f'reference_{k}', v) for k, v in sorted(view['references'].items())]
        for role, p in prompts:
            assert_hooks_clean(adapter._model, baseline)
            text = trainer.chat_prompt(adapter.tokenizer, p)
            inputs = adapter.inputs([text])
            inputs['use_cache'] = True
            with torch.no_grad():
                sequence = adapter._model.generate(**inputs, max_new_tokens=32,
                    do_sample=False, num_beams=1, pad_token_id=adapter.tokenizer.pad_token_id,
                    eos_token_id=adapter.tokenizer.eos_token_id)
            continuation = sequence[0, inputs['input_ids'].shape[1]:].tolist()
            records.append({'family_id': family['id'], 'datatype': family['datatype'],
                'role': role, 'prompt': text, 'continuation_ids': continuation,
                'continuation': adapter.tokenizer.decode(continuation, clean_up_tokenization_spaces=False),
                'semantic_correctness_unverified': True, 'max_new_tokens': 32, 'do_sample': False})
            write_json(out / 'representative_continuations.json', records)
    assert_hooks_clean(adapter._model, baseline)


class WandbLogger:
    def __init__(self, experiment_id, out, config, resume):
        import wandb
        self.wandb = wandb
        path = out / 'wandb.json'
        previous = json.loads(path.read_text()) if resume and path.exists() else {}
        kwargs = {'id': previous['id'], 'resume': 'allow'} if previous.get('id') else {}
        self.run = wandb.init(project=f'default-exp-{experiment_id}',
            name='real32b-set-count-smoke', config=config, **kwargs)
        try:
            from silico.slurm_telemetry import register_wandb_url, register_key_metrics
        except ImportError as exc:
            LOG.warning('Silico telemetry unavailable; W&B logging remains enabled: %s', exc)
        else:
            register_wandb_url()
            register_key_metrics(['eval/total_expected_ce', 'eval/nonnative_primary_ce',
                                  'eval/native_expected_control_ce', 'train/loss'])
        self.step = max(previous.get('last_global_step', -1), getattr(self.run, 'step', 0) - 1)
        self.path = path
        self.save()

    def save(self):
        write_json(self.path, {'id': self.run.id, 'url': self.run.url, 'last_global_step': self.step})

    def log(self, case, local_step, metrics):
        self.step += 1
        # Both aggregate telemetry-friendly keys and case-specific curves.
        payload = {'case': case, 'case_step': local_step, **metrics}
        payload.update({f'{case}/{k}': v for k, v in metrics.items()})
        self.wandb.log(payload, step=self.step)
        self.save()

    def finish(self, exit_code=0):
        self.run.finish(exit_code=exit_code)


def evaluate_fixed(adapter, mask, layers, accepted, hypothesis, bases, trainer):
    result = trainer.evaluate(adapter, mask, layers, accepted, hypothesis, bases)
    return {'summary': aggregate(result['rows']), 'rows': result['rows']}


def cached_clean_baseline(accepted, hypothesis):
    """No inference: the same fixed examples scored from validated clean readouts."""
    rows = []
    for family, caches in accepted:
        for c in caches:
            expected, native = c.meta['expected'][hypothesis], c.meta['native']
            target = 'e' if native == 'd' else 'd'
            other = 'e' if expected == 'd' else 'd'
            lp = c.clean
            top = int(lp.argmax())
            rows.append({'family_id': family['id'], 'view_id': c.meta['id'],
                'config': c.meta['config'], 'donor': c.meta['donor'],
                'expected': expected, 'native': native, 'target': target,
                'expected_ce': float(-lp[c.token_ids[expected]]),
                'expected_correct': top == c.token_ids[expected],
                'native_correct': top == c.token_ids[native],
                'target_correct': top == c.token_ids[target],
                'expected_logodds': float(lp[c.token_ids[expected]]-lp[c.token_ids[other]]),
                'target_native_logodds': float(lp[c.token_ids[target]]-lp[c.token_ids[native]])})
    return {'summary': aggregate(rows), 'rows': rows, 'source': 'cached unpatched recipient readouts'}


def run_case(case_dir, hypothesis, kind, adapter, accepted, layers, bases, trainer,
             torch, logger, protocol_identity, steps, resume=False):
    """Fresh independent optimizer, fixed full evaluation at 0/3/6/.../final."""
    case_dir.mkdir(parents=True, exist_ok=True)
    case = f'{hypothesis}_{kind}'
    baseline = hook_snapshot(adapter._model)
    assert_frozen(adapter._model)
    reset_peak(torch)
    started = time.monotonic()
    shape = (len(layers), adapter.n_heads) + ((adapter.head_dim,) if kind == 'svd' else ())
    mask = torch.nn.Parameter(torch.ones(shape, device=adapter.device, dtype=torch.float32))
    optimizer = torch.optim.AdamW([mask], lr=.01, weight_decay=0.)
    identity = {**protocol_identity, 'hypothesis': hypothesis, 'mask_kind': kind}
    history_path, checkpoint_path = case_dir / 'progress.json', case_dir / 'checkpoint.pt'
    history = {'case': case, 'training': [], 'evaluations': [], 'complete': False,
               'interpretation': 'optimization on selected calibration examples only; no generalization claim'}
    step = 0
    if checkpoint_path.exists():
        if not resume:
            raise ValueError(f'{checkpoint_path} exists; use --resume or a fresh output directory')
        step, loaded_bases = trainer.load_checkpoint(checkpoint_path, identity, mask, optimizer)
        if kind == 'svd':
            if loaded_bases is None:
                raise ValueError('SVD resume checkpoint lacks bases')
            bases = {l: {k: v.to(adapter.device) for k, v in b.items()}
                     for l, b in loaded_bases.items()}
        if not history_path.exists():
            raise ValueError('checkpoint lacks exact initial/history metrics; cannot faithfully resume')
        history = json.loads(history_path.read_text())
        # Progress is written before the checkpoint. Discard a possible interrupted
        # not-yet-committed step and deterministically rerun it from the checkpoint.
        history['training'] = [r for r in history['training'] if r['step'] <= step]
        history['evaluations'] = [r for r in history['evaluations'] if r['step'] <= step]
        history['complete'] = False
    if step > steps:
        raise ValueError('checkpoint exceeds requested total steps')

    def evaluate_at(s):
        assert_hooks_clean(adapter._model, baseline)
        result = evaluate_fixed(adapter, mask, layers, accepted, hypothesis, bases, trainer)
        result['step'] = s
        history['evaluations'].append(result)
        write_json(history_path, history)
        logger.log(case, s, {f'eval/{k}': v for k, v in result['summary'].items() if v is not None})
        assert_hooks_clean(adapter._model, baseline)

    def commit(s):
        write_json(history_path, history)
        trainer.save_checkpoint(checkpoint_path, identity, mask, optimizer, s, bases)

    def gradient_check(gradient):
        if not torch.isfinite(gradient).all():
            raise FloatingPointError('nonfinite mask gradient before optimizer step')
        return gradient

    gradient_handle = mask.register_hook(gradient_check)
    try:
        # Zero transfer should reproduce clean recipient distribution under BOTH
        # head/SVD kernels. Check one representative per family, not just top IDs.
        checks = []
        with torch.no_grad():
            zero = torch.zeros_like(mask)
            for family, caches in accepted:
                c = caches[0]
                with trainer.patched(adapter, zero, layers, c.q_src, bases):
                    actual = adapter.logprobs([c.destination])[0].cpu()
                clean = c.clean
                torch.testing.assert_close(actual, clean, rtol=0., atol=.0625)
                if int(actual.argmax()) != int(clean.argmax()):
                    raise AssertionError('zero-mask changed clean argmax')
                checks.append({'family_id': family['id'], 'max_abs_logprob_error': float((actual-clean).abs().max()),
                               'top_id_equal': True, 'atol': .0625, 'rtol': 0.})
        write_json(case_dir / 'zero_mask_checks.json', checks)
        if not history['evaluations']:
            if step:
                raise ValueError('missing initial evaluation on resume')
            evaluate_at(0)
        if step == 0 and not (case_dir / 'initial_checkpoint.pt').exists():
            trainer.save_checkpoint(case_dir / 'initial_checkpoint.pt', identity, mask, optimizer, 0, bases)
        if (step % 3 == 0 or step == steps) and history['evaluations'][-1]['step'] != step:
            evaluate_at(step)
        commit(step)
        for current in range(step, steps):
            assert_hooks_clean(adapter._model, baseline)
            family, caches = accepted[current % len(accepted)]
            synchronize(torch)
            start_step = time.monotonic()
            metrics = trainer.train_step(adapter, mask, optimizer, layers, caches, hypothesis,
                bases=bases, kl_weight=0., sparsity=0., control_weight=1.)
            synchronize(torch)
            if mask.grad is None or not torch.isfinite(mask.grad).all():
                raise FloatingPointError('missing or nonfinite accumulated gradient')
            if not torch.isfinite(mask).all() or not bool(((mask >= 0) & (mask <= 1)).all()):
                raise FloatingPointError('mask nonfinite or outside [0,1]')
            if any(not math.isfinite(v) for v in metrics.values()):
                raise FloatingPointError('nonfinite training metric')
            assert_frozen(adapter._model)
            assert_hooks_clean(adapter._model, baseline)
            s = current + 1
            record = {'step': s, 'family_id': family['id'], **metrics,
                      'lr': optimizer.param_groups[0]['lr'],
                      'mask_min': float(mask.detach().min()), 'mask_max': float(mask.detach().max()),
                      'mask_mean_delta_from_initial': float((mask.detach()-1).mean()),
                      'mask_l2_delta_from_initial': float((mask.detach()-1).norm()),
                      'seconds': time.monotonic() - start_step}
            history['training'].append(record)
            # Save every optimizer step BEFORE expensive optional evaluation.
            commit(s)
            logger.log(case, s, {f'train/{k}': v for k, v in record.items() if isinstance(v, (float, int))})
            if s % 3 == 0 or s == steps:
                evaluate_at(s)
                commit(s)
        if history['evaluations'][-1]['step'] != steps:
            evaluate_at(steps)
        synchronize(torch)
        initial, final = history['evaluations'][0], history['evaluations'][-1]
        history.update(complete=True, initial=initial, final=final,
            signed_change_final_minus_initial=signed_changes(initial['summary'], final['summary']),
            loss_descent_observed=final['summary']['total_expected_ce'] < initial['summary']['total_expected_ce'],
            total_steps=steps, elapsed_seconds=time.monotonic() - started,
            **memory(torch))
        commit(steps)
        write_json(case_dir / 'summary.json', {k: v for k, v in history.items() if k not in ('training', 'evaluations')})
        return history
    except BaseException:
        write_json(case_dir / 'failure.json', {'case': case, 'traceback': traceback.format_exc(),
            'last_saved_checkpoint': str(checkpoint_path), 'elapsed_seconds': time.monotonic()-started,
            **memory(torch)})
        raise
    finally:
        gradient_handle.remove()
        remove_new_hooks(adapter._model, baseline)
        assert_hooks_clean(adapter._model, baseline)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--project-root', required=True)
    p.add_argument('--examples-json', default=str(Path(__file__).resolve().parents[2] / 'set_count_design/concrete_examples.json'))
    p.add_argument('--steps', type=int, default=12, help='total steps per independent case, including resume')
    p.add_argument('--layers', default='20..55', help='inclusive range or comma-separated indices')
    p.add_argument('--max-length', type=int, default=2048)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--experiment-id', default=os.environ.get('SILICO_EXPERIMENT_ID'))
    p.add_argument('--stage-manifest', help='parent-produced non-secret provenance JSON, hashed and recorded')
    p.add_argument('--resume', action='store_true', help='resume trusted checkpoints in this output directory')
    return p


def execute(args, out):
    import torch
    import transformers
    model_dir = Path(args.model_dir).resolve()
    project_root = Path(args.project_root).resolve()
    examples_path = Path(args.examples_json).resolve()
    layers = parse_layers(args.layers)
    if not args.experiment_id:
        raise ValueError('set SILICO_EXPERIMENT_ID or provide --experiment-id for W&B')
    if args.steps < 0 or args.steps > 48:
        raise ValueError('--steps must be in [0,48]; this is a bounded smoke, not a large training run')
    if not args.resume and (out / 'protocol.json').exists():
        raise ValueError('output already contains a protocol; use a fresh directory or --resume')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('requires exactly one visible CUDA GPU')
    sys.path.insert(0, str(project_root))
    from src.diff import set_count_data as data, set_count_train as trainer
    from src.diff.query_mask_ops import svd_bases
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    reset_peak(torch)
    startup = time.monotonic()
    sources = [project_root / 'src/diff' / name for name in
               ('set_count_data.py', 'set_count_train.py', 'query_mask_ops.py', 'nrgs_prefix.py', 'data.py')]
    sources += [project_root / 'src/goals/pivot_token_filter.py', Path(__file__), Path(__file__).with_name('smoke_helpers.py')]
    pool_path = project_root / data.N.PIVOT_POOL.format(model=MODEL.split('/')[-1])
    candidates = build_candidates(project_root, examples_path, data, args.seed)
    write_json(out / 'candidate_families.json', candidates)
    protocol = {'model': MODEL, 'requested_revision': REVISION, 'model_dir': str(model_dir),
        'config': vars(args), 'layers': layers, 'dtype': 'bfloat16', 'attention': 'eager',
        'single_device': 'cuda:0', 'cases': [list(c) for c in CASES],
        'candidate_order': {dt: [f['id'] for f in fs] for dt, fs in candidates.items()},
        'selection': 'first whole nine-view passing family per datatype, at most four candidates each; before training',
        'objective': 'equal-family fixed-collection expected-token CE; equal nine views per family',
        'optimizer': {'name': 'AdamW', 'lr': .01, 'weight_decay': 0., 'mask_init': 1.,
                      'control_weight': 1., 'kl_weight': 0., 'sparsity': 0.},
        'evaluation_steps': sorted(set([0, *range(3, args.steps + 1, 3), args.steps])),
        'training_schedule': 'one complete family per step; cycle in str,int,list[int] accepted order',
        'scope': 'train-only calibration; original bank explored; optimization only, no heldout/generalization claims',
        'semantic_correctness_unverified': True, 'loss_descent_required_for_exit': False,
        'source_hashes': {(str(path.relative_to(project_root)) if path.is_relative_to(project_root)
                           else 'runner/' + path.name): sha(path) for path in sources},
        'pool_sha256': sha(pool_path), 'examples_sha256': sha(examples_path),
        'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__,
        'libraries': {name: importlib.metadata.version(name) for name in ('tokenizers', 'safetensors', 'wandb')},
        'cuda_version': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
        'gpu_total_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
        'model_file_hashes': {path.name: sha(path) for path in model_dir.iterdir()
            if path.is_file() and path.name in ('config.json', 'generation_config.json', 'model.safetensors.index.json',
                'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'vocab.json', 'merges.txt')}}
    if args.stage_manifest:
        path = Path(args.stage_manifest).resolve()
        protocol['stage_manifest'] = {'path': str(path), 'sha256': sha(path), 'contents': json.loads(path.read_text())}
    write_json(out / ('protocol_resume_attempt.json' if args.resume else 'protocol.json'), protocol)
    logger = WandbLogger(args.experiment_id, out, protocol, args.resume)
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True,
            trust_remote_code=False)
        model = transformers.AutoModelForCausalLM.from_pretrained(str(model_dir), local_files_only=True,
            trust_remote_code=False, dtype=torch.bfloat16, attn_implementation='eager').to('cuda:0')
        Adapter = make_adapter_class(trainer, torch)
        adapter = Adapter(model, tokenizer, max_length=args.max_length)
        if (adapter.n_layer, adapter.n_heads, adapter.n_kv_heads, adapter.head_dim) != (64, 40, 8, 128):
            raise ValueError('staged model has wrong architecture geometry')
        if {p.dtype for p in model.parameters()} != {torch.bfloat16}:
            raise ValueError('base parameters are not uniformly BF16')
        if getattr(model.config, '_attn_implementation', None) != 'eager':
            raise ValueError('base model did not select eager attention')
        assert_frozen(model)
        baseline = hook_snapshot(model)
        protocol.update(resolved_revision=getattr(model.config, '_commit_hash', None),
            tokenizer_revision=getattr(tokenizer, 'init_kwargs', {}).get('_commit_hash'),
            model_config=model.config.to_dict(), tokenizer_class=type(tokenizer).__name__,
            model_class=type(model).__name__, parameter_count=sum(p.numel() for p in model.parameters()),
            revision_caveat='local_files_only load: requested revision is not proof of staged weight identity; consult stage manifest',
            base_frozen=True, base_no_grad=True)
        probe = trainer.chat_prompt(tokenizer, candidates['str'][0]['views'][0]['destination'])
        with torch.no_grad():
            ordinary = trainer.HFAdapter.logprobs(adapter, [probe])
            optimized = adapter.logprobs([probe])
        torch.testing.assert_close(optimized, ordinary, rtol=0., atol=.0625)
        if not torch.equal(ordinary.argmax(-1), optimized.argmax(-1)):
            raise AssertionError('logits_to_keep changed greedy last-token prediction')
        protocol['last_token_equivalence'] = {'max_abs_logprob_error': float((ordinary-optimized).abs().max()),
            'atol': .0625, 'rtol': 0., 'top_id_equal': True}
        del ordinary, optimized
        synchronize(torch)
        protocol['startup_seconds'] = time.monotonic() - startup
        protocol['startup_memory'] = memory(torch)
        write_json(out / 'protocol.json', protocol)
        diagnostics = []
        def on_attempt(family, accepted, record, attempts, rejected):
            diagnostics.append({'family_id': family['id'], 'accepted': bool(accepted),
                                'readouts': readout_diagnostics(family, adapter, trainer)})
            write_json(out / 'selection_attempts.json', attempts)
            write_json(out / 'rejected_families.json', rejected)
            write_json(out / 'validation_readouts.json', diagnostics)
        accepted, rejected, attempts = select_first(candidates,
            lambda fs: trainer.validate_and_cache(adapter, fs, layers, smoke=False), on_attempt)
        assert_hooks_clean(model, baseline)
        write_json(out / 'selected_families.json', [f for f, _ in accepted])
        write_json(out / 'selected_prompts.json', [{'family_id': f['id'], 'views': [
            {'view_id': c.meta['id'], 'destination': c.destination,
             'source': trainer.chat_prompt(tokenizer, c.meta['source']),
             'references': {k: trainer.chat_prompt(tokenizer, p) for k, p in c.meta['references'].items()},
             'token_ids': c.token_ids, 'validation': c.validation} for c in caches]} for f, caches in accepted])
        missing = [dt for dt in DATATYPES if not any(f['datatype'] == dt for f, _ in accepted)]
        protocol['missing_datatypes'] = missing
        protocol['accepted_family_ids'] = [f['id'] for f, _ in accepted]
        protocol['validation_seconds'] = time.monotonic() - startup - protocol['startup_seconds']
        write_json(out / 'protocol.json', protocol)
        if not accepted:
            write_json(out / 'summary.json', {'status': 'no_authentic_accepted_families', 'trained': False,
                'missing_datatypes': missing, 'semantic_correctness_unverified': True})
            raise RuntimeError('No authentic nine-view family passes next-token validation; no training performed')
        # Selection frozen permanently before generation, SVD construction, or optimization.
        adapter.audit_enabled = False
        adapter.readout_audit.clear()
        trainer.atomic_save(out / 'validation_cache.pt', {'layers': layers, 'families': accepted})
        write_json(out / 'clean_baselines.json', {hyp: cached_clean_baseline(accepted, hyp)
                                                  for hyp in ('set', 'count')})
        generate_examples(adapter, accepted, trainer, torch, baseline, out)
        identity = {'model': MODEL, 'revision': REVISION, 'layers': layers, 'seed': args.seed,
            'dtype': 'bfloat16', 'max_length': args.max_length,
            'source_hashes': protocol['source_hashes'], 'model_file_hashes': protocol['model_file_hashes'],
            'pool_sha256': protocol['pool_sha256'], 'optimizer': protocol['optimizer'],
            'selected_fingerprint': data.digest([f for f, _ in accepted]),
            'readout_fingerprint': data.digest([[c.meta['id'], c.token_ids] for _, cs in accepted for c in cs])}
        results = {}
        bases = None
        for hypothesis, kind in CASES:
            assert_hooks_clean(model, baseline)
            if kind == 'svd' and bases is None:
                reset_peak(torch)
                started = time.monotonic()
                bases = svd_bases(adapter, layers)
                if any(not torch.isfinite(v).all() for b in bases.values() for v in b.values()):
                    raise FloatingPointError('nonfinite SVD bases')
                synchronize(torch)
                write_json(out / 'svd_setup.json', {'seconds': time.monotonic()-started, **memory(torch),
                    'method': 'qk-thin-qr-svd; pre-RoPE; original implementation unchanged'})
            result = run_case(out / f'{hypothesis}_{kind}', hypothesis, kind, adapter, accepted,
                layers, bases if kind == 'svd' else None, trainer, torch, logger, identity, args.steps, args.resume)
            results[f'{hypothesis}_{kind}'] = {k: v for k, v in result.items() if k not in ('training', 'evaluations')}
            assert_frozen(model)
            assert_hooks_clean(model, baseline)
            write_json(out / 'summary.json', {'status': 'complete' if len(results) == 4 else 'in_progress',
                'cases': results, 'missing_datatypes': missing,
                'conclusion_scope': 'optimization on fixed train-only calibration examples; inspect primary/control separately',
                'semantic_correctness_unverified': True})
        logger.finish()
    except BaseException:
        try:
            logger.finish(exit_code=1)
        except Exception:
            LOG.warning('W&B shutdown failed while preserving the original error', exc_info=True)
        raise


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    args = parser().parse_args(argv)
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        execute(args, out)
    except BaseException:
        diagnostic = {'status': 'failed', 'traceback': traceback.format_exc(),
                      'elapsed_seconds': time.monotonic() - started,
                      'requested_revision': REVISION, 'model_dir': str(Path(args.model_dir).resolve())}
        torch = sys.modules.get('torch')
        if torch is not None and torch.cuda.is_initialized():
            try:
                diagnostic.update(memory(torch))
            except Exception:
                LOG.warning('CUDA failure-memory accounting unavailable', exc_info=True)
        write_json(out / 'failure.json', diagnostic)
        LOG.exception('Real-model smoke failed; diagnostic and completed progress preserved at %s', out)
        raise


if __name__ == '__main__':
    main()
