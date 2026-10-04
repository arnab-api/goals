"""Standalone, single-device Qwen2 query-mask training.

Each invocation trains ONE hypothesis, with no overlap/orthogonality coupling.
Validation here is next-token-only. --allow-next-token-only is mandatory for
non-smoke optimization; no full semantic validation is claimed. Test data is
never used for optimization or selection. --max-steps is a TOTAL step ceiling.
"""
from __future__ import annotations

import argparse
import hashlib
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import platform
import random
import time

import torch
import torch.nn.functional as F

from src.diff.query_mask_ops import q_blend_hooks, svd_bases, svd_blend_hooks
from src.diff.set_count_data import digest, meaningful_contrast, verify_dataset

SYSTEM = ('You are a helpful assistant proficient in programming. '
          'Always wrap your code inside <code></code> blocks. Just the code, no need to comment or explain.')


class ByteTokenizer:
    """Deterministic offline fixture ONLY for random-model mechanics checks."""
    pad_token_id = 0
    eos_token_id = 1
    padding_side = 'left'
    vocab_size = 261

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs['continue_final_message'] and not kwargs['add_generation_prompt']
        return ''.join('<' + m['role'] + '>\n' + m['content'] +
                       ('\n' if i < len(messages) - 1 else '')
                       for i, m in enumerate(messages))

    def __call__(self, prompts, **kwargs):
        ids = [[x + 3 for x in p.encode('utf-8')] for p in prompts]
        width = max(map(len, ids))
        return {'input_ids': torch.tensor([[0] * (width-len(x)) + x for x in ids]),
                'attention_mask': torch.tensor([[0] * (width-len(x)) + [1]*len(x) for x in ids])}

    def decode(self, ids, **kwargs):
        return bytes([x-3 for x in ids if 3 <= x <= 258]).decode('utf-8', errors='replace')


def chat_prompt(tokenizer, prompt):
    code = prompt['code']
    text = tokenizer.apply_chat_template(
        [{'role': 'system', 'content': SYSTEM},
         {'role': 'user', 'content': prompt['question']},
         {'role': 'assistant', 'content': code}],
        add_generation_prompt=False, continue_final_message=True, tokenize=False,
        chat_template_kwargs={'enable_thinking': False})
    # Some HF releases rstrip the final assistant content. Restore only its
    # original trailing whitespace, never strip the informative expression.
    if not text.endswith(code):
        trimmed = code.rstrip()
        if trimmed and text.endswith(trimmed):
            text += code[len(trimmed):]
        else:
            raise ValueError('chat template did not preserve assistant prefill')
    return text


class HFAdapter:
    def __init__(self, model, tokenizer, max_length=2048):
        c = model.config
        if c.model_type != 'qwen2':
            raise ValueError('Only Qwen2 architecture is supported')
        self._model = model.eval()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.n_heads = c.num_attention_heads
        self.n_kv_heads = c.num_key_value_heads
        self.n_layer = c.num_hidden_layers
        self.head_dim = getattr(c, 'head_dim', None) or c.hidden_size // self.n_heads
        self.attn_module_name_format = 'model.layers.{}.self_attn'
        if self.n_heads % self.n_kv_heads or self.n_heads * self.head_dim != c.hidden_size:
            raise ValueError('unsupported GQA dimensions')
        devices = {p.device for p in model.parameters()}
        if len(devices) != 1 or next(iter(devices)).type == 'meta':
            raise ValueError('one fully materialized device required; no automatic sharding')
        self.device = next(iter(devices))
        if getattr(model, 'is_gradient_checkpointing', False):
            raise ValueError('gradient checkpointing is not supported')
        for layer in range(self.n_layer):
            q = model.get_submodule(self.attn_module_name_format.format(layer) + '.q_proj')
            k = model.get_submodule(self.attn_module_name_format.format(layer) + '.k_proj')
            if q.out_features != self.n_heads*self.head_dim or k.out_features != self.n_kv_heads*self.head_dim:
                raise ValueError('unexpected q/k projection dimensions')
        for p in model.parameters():
            p.requires_grad_(False)
        tokenizer.padding_side = 'left'
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    def inputs(self, prompts):
        values = self.tokenizer(prompts, return_tensors='pt', padding=True,
                                truncation=False, add_special_tokens=False)
        if values['input_ids'].shape[1] > self.max_length:
            raise ValueError(f'input exceeds max_length={self.max_length}; never silently truncate')
        ids = values['input_ids'].to(self.device)
        attention = values['attention_mask'].to(self.device)
        positions = attention.long().cumsum(-1)-1
        positions.masked_fill_(attention == 0, 0)
        return dict(input_ids=ids, attention_mask=attention, position_ids=positions, use_cache=False)

    def logprobs(self, prompts):
        return self._model(**self.inputs(prompts)).logits[:, -1].float().log_softmax(-1)

    @torch.no_grad()
    def capture(self, prompts, layers):
        captured, handles = {}, []
        def hook_for(layer):
            def hook(module, inputs, output):
                o = output[0] if isinstance(output, tuple) else output
                captured[layer] = o[:, -1].detach().float().cpu()
            return hook
        try:
            for layer in layers:
                mod = self._model.get_submodule(self.attn_module_name_format.format(layer) + '.q_proj')
                handles.append(mod.register_forward_hook(hook_for(layer)))
            self.logprobs(prompts)
        finally:
            for handle in handles:
                handle.remove()
        return torch.stack([captured[layer] for layer in layers], dim=1)


@contextmanager
def patched(adapter, mask, layers, q_src, bases=None):
    handles = (q_blend_hooks(adapter, mask, layers, q_src) if bases is None else
               svd_blend_hooks(adapter, mask, layers, bases, q_src))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@dataclass
class ViewCache:
    meta: dict
    destination: str
    q_src: torch.Tensor
    token_ids: dict
    references: dict
    clean: torch.Tensor
    validation: dict


@torch.no_grad()
def validate_and_cache(adapter, families, layers, *, smoke=False):
    accepted, rejected = [], []
    for family in families:
        caches, errors = [], []
        # In-memory deduplication only; no persistent stale-cache path.
        qs, lps = {}, {}
        for view in family['views']:
            try:
                src = chat_prompt(adapter.tokenizer, view['source'])
                dst = chat_prompt(adapter.tokenizer, view['destination'])
                refs = {label: chat_prompt(adapter.tokenizer, p)
                        for label, p in view['references'].items()}
                if src not in qs:
                    qs[src] = adapter.capture([src], layers)
                for prompt in set([dst, *refs.values()]):
                    if prompt not in lps:
                        lps[prompt] = adapter.logprobs([prompt])[0].cpu()
                tokens = {label: int(lps[prompt].argmax()) for label, prompt in refs.items()}
                decoded = {k: adapter.tokenizer.decode([v], clean_up_tokenization_spaces=False)
                           for k, v in tokens.items()}
                validation = {'level': 'mechanics_only' if smoke else 'next_token_only',
                              'reference_top_ids': tokens.copy(), 'reference_top_strings': decoded,
                              'stored_token_agreement': {k: decoded[k] == view['pool_tokens'][k] for k in tokens},
                              'readout_roles': view['readout_roles'], 'block': view['block'],
                              'source_completion_verified': False, 'full_completion_verified': False}
                if smoke:
                    # Dedicated fixture outputs are deliberately unrelated to a
                    # random model's top token. These are NOT semantic labels.
                    tokens = dict(zip(view['readout_roles'], (259, 260)))
                    validation['artificial_readout_ids'] = tokens
                else:
                    left, right = view['readout_roles']
                    if tokens[left] == tokens[right]:
                        raise ValueError('reference next tokens coincide')
                    if not meaningful_contrast(view['prefix'], decoded[left], decoded[right]):
                        raise ValueError('reference contrast fails contextual lexical policy')
                    if int(lps[dst].argmax()) != tokens[view['native']]:
                        raise ValueError('clean destination disagrees with native reference')
                caches.append(ViewCache(view, dst, qs[src], tokens,
                                        {k: lps[p] for k, p in refs.items()}, lps[dst], validation))
            except ValueError as exc:
                errors.append({'view_id': view['id'], 'reason': str(exc)})
        if errors or not caches:
            # Keep whole matched families, not a convenient subset of donors.
            rejected.append({'family_id': family['id'], 'split': family['split'],
                             'reasons': errors or [{'reason': 'empty family'}]})
        else:
            accepted.append((family, caches))
    return accepted, rejected


def expected_loss(log_p, token_id, log_reference, kl_weight=0.):
    """Absolute expected-token CE, also for native-expected controls.

    This penalizes a third token winning, unlike a two-token margin alone.
    KL has orientation P_patched || P_expected_reference.
    """
    ce = -log_p[token_id]
    kl = F.kl_div(log_reference.to(log_p.device), log_p, reduction='sum', log_target=True)
    return ce + kl_weight * kl


def view_weights(caches, hypothesis=None, control_weight=1.):
    """Equal blocks, then configurations, donors and naming variants.

    Native-control weighting does not renormalize primary contributions. The
    9-view de block and 4-view da block each get half the default family mass.
    """
    if control_weight < 0:
        raise ValueError('control_weight must be nonnegative')
    blocks = sorted({c.meta.get('block', 'de') for c in caches})
    weights = []
    for c in caches:
        block = c.meta.get('block', 'de')
        same_block = [x for x in caches if x.meta.get('block', 'de') == block]
        configs = {x.meta['config'] for x in same_block}
        same = [x for x in same_block if x.meta['config'] == c.meta['config']]
        donors = {x.meta['donor'] for x in same}
        n = sum(x.meta['donor'] == c.meta['donor'] for x in same)
        factor = (control_weight if hypothesis is not None and
                  c.meta['expected'][hypothesis] == c.meta['native'] else 1.)
        weights.append(factor / (len(blocks)*len(configs)*len(donors)*n))
    return weights


def train_step(adapter, mask, optimizer, layers, caches, hypothesis, *, bases=None,
               kl_weight=0., sparsity=0., control_weight=1.):
    optimizer.zero_grad(set_to_none=True)
    total = 0.
    for c, weight in zip(caches, view_weights(caches, hypothesis, control_weight)):
        if weight == 0:
            continue
        label = c.meta['expected'][hypothesis]
        with patched(adapter, mask, layers, c.q_src, bases):
            lp = adapter.logprobs([c.destination])[0]
            loss = weight * expected_loss(lp, c.token_ids[label], c.references[label], kl_weight)
            loss.backward()
        total += float(loss.detach())
    sparse = sparsity * mask.mean()
    sparse.backward()
    grad_norm = float(mask.grad.norm())
    optimizer.step()
    with torch.no_grad():
        mask.clamp_(0., 1.)
    return {'loss': total + float(sparse.detach()), 'gradient_norm': grad_norm,
            'mask_mean': float(mask.detach().mean())}


@torch.no_grad()
def evaluate(adapter, mask, layers, families, hypothesis, bases=None):
    rows, whole = [], []
    for family, caches in families:
        correct = []
        for c, weight in zip(caches, view_weights(caches)):
            expected = c.meta['expected'][hypothesis]
            native = c.meta['native']
            target = next(k for k in c.token_ids if k != native)
            with patched(adapter, mask, layers, c.q_src, bases):
                lp = adapter.logprobs([c.destination])[0]
            top = int(lp.argmax())
            other = next(k for k in c.token_ids if k != expected)
            ok = top == c.token_ids[expected]; correct.append(ok)
            rows.append({'family_id': family['id'], 'view_id': c.meta['id'],
                         'split': family['split'], 'datatype': family['datatype'],
                         'block': c.meta.get('block', 'de'), 'view_weight': weight,
                         'prefix': c.meta.get('prefix'), 'readout_roles': list(c.token_ids),
                         'src_order': c.meta.get('src_order'), 'src_done': c.meta.get('src_done'),
                         'dst_order': c.meta.get('dst_order'), 'dst_done': c.meta.get('dst_done'),
                         'naming_variant': c.meta.get('naming_variant'),
                         'diagnostic_pair_group': c.meta.get('diagnostic_pair_group'),
                         'config': c.meta['config'],
                         'donor': c.meta['donor'], 'hypothesis': hypothesis,
                         'expected': expected, 'native': native, 'target': target,
                         'expected_correct': ok, 'native_correct': top == c.token_ids[native],
                         'target_correct': top == c.token_ids[target], 'top_id': top,
                         'expected_ce': float(-lp[c.token_ids[expected]]),
                         'expected_logodds': float(lp[c.token_ids[expected]]-lp[c.token_ids[other]]),
                         'target_native_logodds': float(lp[c.token_ids[target]]-lp[c.token_ids[native]]),
                         'clean_top_id': int(c.clean.argmax()), 'validation': c.validation})
        whole.append(all(correct))
    return {'n_families': len(whole), 'n_views': len(rows),
            'whole_family_accuracy': sum(whole)/len(whole) if whole else None,
            'expected_accuracy': sum(r['view_weight']*r['expected_correct'] for r in rows)/len(whole) if whole else None,
            'unweighted_expected_accuracy': sum(r['expected_correct'] for r in rows)/len(rows) if rows else None,
            'rows': rows}


def atomic_save(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    torch.save(payload, tmp)
    os.replace(tmp, path)


def save_checkpoint(path, identity, mask, optimizer, step, bases=None):
    atomic_save(path, {'schema': 1, 'identity': identity, 'mask': mask.detach().cpu(),
                      'optimizer': optimizer.state_dict(), 'step': step,
                      'bases': ({l: {k: v.cpu() for k, v in b.items()} for l, b in bases.items()}
                                if bases is not None else None),
                      'basis_metadata': {'method': 'qk-thin-qr-svd', 'includes_rope': False},
                      'rng': {'python': random.getstate(), 'torch': torch.get_rng_state(),
                              'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}})


def load_checkpoint(path, identity, mask, optimizer):
    # Only load a checkpoint produced by this script / a trusted researcher.
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state.get('schema') != 1 or state['identity'] != identity:
        raise ValueError('resume identity mismatch (hypothesis/model/data/layers/optimizer/settings)')
    if state['mask'].shape != mask.shape:
        raise ValueError('resume mask shape mismatch')
    with torch.no_grad():
        mask.copy_(state['mask'].to(mask.device))
    optimizer.load_state_dict(state['optimizer'])
    random.setstate(state['rng']['python']); torch.set_rng_state(state['rng']['torch'])
    if state['rng']['cuda']:
        if not torch.cuda.is_available():
            raise ValueError('CUDA checkpoint cannot exactly resume on CPU')
        torch.cuda.set_rng_state_all(state['rng']['cuda'])
    return state['step'], state['bases']


def load_model(args):
    # Lazy imports keep the builder usable without transformers.
    from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
    if args.smoke_random:
        tokenizer = ByteTokenizer()
        config = Qwen2Config(vocab_size=tokenizer.vocab_size, hidden_size=32,
                            intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                            num_key_value_heads=2, max_position_embeddings=args.max_length,
                            attention_dropout=0., pad_token_id=0, eos_token_id=1,
                            bos_token_id=2, tie_word_embeddings=False)
        config._attn_implementation = 'eager'
        # Keep the offline fixture in float32 regardless of --dtype, which
        # controls pretrained loading only. Record the actual dtype below.
        model = Qwen2ForCausalLM(config).to(args.device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision,
                                                 trust_remote_code=False)
        dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
                 'float16': torch.float16}[args.dtype]
        model = AutoModelForCausalLM.from_pretrained(
            args.model, revision=args.revision, torch_dtype=dtype,
            trust_remote_code=False, attn_implementation='eager').to(args.device)
    return HFAdapter(model, tokenizer, args.max_length)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--hypothesis', choices=['set', 'count'], required=True)
    p.add_argument('--model', default='Qwen/Qwen2.5-Coder-32B-Instruct',
                   help='pretrained target; ignored by --smoke-random (offline random Qwen2)')
    p.add_argument('--revision', default='main')
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--dtype', choices=['float32', 'bfloat16', 'float16'], default='bfloat16',
                   help='pretrained weight dtype; random smoke always uses float32')
    p.add_argument('--mask-kind', choices=['head', 'svd'], default='head')
    p.add_argument('--mask-init', type=float, default=1.,
                   help='initial mask value in [0,1]; default preserves full-transfer initialization')
    p.add_argument('--layers', help='comma-separated layer indices; default all')
    p.add_argument('--max-length', type=int, default=2048)
    p.add_argument('--max-families', type=int, default=8, help='cap PER split, selected by stable family ID')
    p.add_argument('--max-steps', type=int, default=10, help='TOTAL optimizer-step ceiling, including resumed steps')
    p.add_argument('--lr', type=float, default=.01)
    p.add_argument('--kl-weight', type=float, default=0.)
    p.add_argument('--sparsity', type=float, default=0.)
    p.add_argument('--control-weight', type=float, default=1.,
                   help='native-expected view loss multiplier; 0 is a primary-only ablation')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--resume', help='trusted checkpoint path')
    p.add_argument('--smoke-random', action='store_true')
    p.add_argument('--allow-next-token-only', action='store_true')
    p.add_argument('--validate-only', action='store_true')
    p.add_argument('--evaluate-test', action='store_true', help='explicit final test evaluation, never selection')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.max_steps < 0 or args.max_families < 1:
        raise ValueError('max_steps must be nonnegative and max_families positive')
    if not 0. <= args.mask_init <= 1.:
        raise ValueError('mask_init must be in [0,1]')
    if args.lr <= 0 or min(args.kl_weight, args.sparsity, args.control_weight) < 0:
        raise ValueError('lr must be positive; loss weights must be nonnegative')
    if not (args.validate_only or args.smoke_random or args.allow_next_token_only):
        raise ValueError('Full semantic validation is not implemented. Use --validate-only or explicitly opt into --allow-next-token-only exploratory training.')
    started = time.monotonic()
    random.seed(args.seed); torch.manual_seed(args.seed)
    dataset = verify_dataset(json.loads(Path(args.dataset).read_text()))
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    if (out/'checkpoint.pt').exists() and (not args.resume or args.validate_only):
        raise ValueError('output already has a checkpoint; use a fresh directory for validation or --resume for training')
    adapter = load_model(args)
    layers = list(range(adapter.n_layer)) if not args.layers else [int(x) for x in args.layers.split(',')]
    if not layers or len(set(layers)) != len(layers) or any(l < 0 or l >= adapter.n_layer for l in layers):
        raise ValueError('invalid or repeated layers')
    splits = ['train', 'dev'] + (['test'] if args.evaluate_test else [])
    chosen = []
    for split in splits:
        chosen += sorted([f for f in dataset['families'] if f['split'] == split],
                         key=lambda f: f['id'])[:args.max_families]
    accepted, rejected = validate_and_cache(adapter, chosen, layers, smoke=args.smoke_random)
    validation = [{'family_id': f['id'], 'split': f['split'],
                   'views': [{'view_id': c.meta['id'], **c.validation} for c in cs]}
                  for f, cs in accepted]
    grouped = {s: [(f, cs) for f, cs in accepted if f['split'] == s] for s in splits}
    import transformers
    actual_model = 'random-Qwen2-mechanics-fixture' if args.smoke_random else args.model
    actual_revision = None if args.smoke_random else args.revision
    actual_dtype = str(next(adapter._model.parameters()).dtype)
    code_hashes = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                   for name in ['set_count_train.py', 'query_mask_ops.py']}
    split_provenance = {
        s: {'considered': sum(f['split'] == s for f in chosen),
            'accepted': len(grouped.get(s, [])),
            'rejected': sum(f['split'] == s for f in rejected),
            'evaluated': s in splits,
            'reason': ('bounded per-split selection; whole-family validation' if s in splits
                       else 'test not requested; never used for selection')}
        for s in ['train', 'dev', 'test']}
    runtime = {'config': vars(args), 'python': platform.python_version(), 'torch': torch.__version__,
               'actual_model': actual_model, 'actual_revision': actual_revision,
               'actual_dtype': actual_dtype, 'code_sha256': code_hashes,
               'split_provenance': split_provenance,
               'rejected_family_reasons_path': 'rejected_families.json',
               'smoke_dtype_policy': 'float32 irrespective of --dtype' if args.smoke_random else None,
               'base_frozen': all(not p.requires_grad for p in adapter._model.parameters()),
               'base_eval_mode': not adapter._model.training,
               'max_families_semantics': 'cap per split, not full-dataset training',
               'transformers': transformers.__version__,
               'device': str(adapter.device), 'dataset_fingerprint': dataset['fingerprint'],
               'validation_level': 'mechanics_only' if args.smoke_random else 'next_token_only',
               'full_completion_verified': False, 'source_completion_verified': False,
               'accepted_families': {s: len(v) for s, v in grouped.items()},
               'rejected_families': len(rejected), 'scientific_score': False if args.smoke_random else 'exploratory_only'}
    def publish_preflight():
        (out/'rejected_families.json').write_text(json.dumps(rejected, indent=2)+'\n')
        (out/'validation.json').write_text(json.dumps(validation, indent=2)+'\n')
        (out/'runtime.json').write_text(json.dumps(runtime, indent=2)+'\n')

    if not accepted or (not args.validate_only and not grouped['train']):
        # Never replace the provenance of an existing run on a failed resume.
        if not args.resume:
            publish_preflight()
        raise ValueError('zero accepted families for requested operation; no existing resume artifacts were changed')
    if args.validate_only:
        publish_preflight()
        print(json.dumps(runtime)); return runtime
    bases = svd_bases(adapter, layers) if args.mask_kind == 'svd' else None
    shape = (len(layers), adapter.n_heads) + ((adapter.head_dim,) if bases is not None else ())
    mask = torch.nn.Parameter(torch.full(shape, args.mask_init, device=adapter.device))
    optimizer = torch.optim.AdamW([mask], lr=args.lr, weight_decay=0.)
    identity = {'hypothesis': args.hypothesis, 'model': actual_model, 'revision': actual_revision,
                'code_sha256': code_hashes,
                'resolved_revision': getattr(adapter._model.config, '_commit_hash', None),
                'model_config': digest(adapter._model.config.to_dict()),
                'tokenizer_revision': getattr(adapter.tokenizer, 'init_kwargs', {}).get('_commit_hash'),
                'dataset_fingerprint': dataset['fingerprint'], 'layers': layers,
                'mask_kind': args.mask_kind, 'mask_init': args.mask_init,
                'smoke_random': args.smoke_random,
                'accepted_train_ids': [f['id'] for f, _ in grouped['train']],
                'training_readout_fingerprint': digest([
                    [c.meta['id'], c.token_ids] for _, cs in grouped['train'] for c in cs]),
                'lr': args.lr, 'weight_decay': 0., 'kl_weight': args.kl_weight,
                'sparsity': args.sparsity, 'control_weight': args.control_weight,
                'seed': args.seed, 'dtype': str(next(adapter._model.parameters()).dtype),
                'max_length': args.max_length, 'allow_next_token_only': args.allow_next_token_only,
                'objective_version': 'family-block-balanced-expected-ce-v2'}
    step = 0
    if args.resume:
        step, loaded_bases = load_checkpoint(args.resume, identity, mask, optimizer)
        if bases is not None:
            if loaded_bases is None:
                raise ValueError('missing saved SVD bases')
            bases = {l: {k: v.to(adapter.device) for k, v in b.items()} for l, b in loaded_bases.items()}
    if step > args.max_steps:
        raise ValueError('checkpoint already exceeds total --max-steps')
    # Identity and total-step checks must succeed before publishing new metadata.
    publish_preflight()
    history = []
    while step < args.max_steps:
        # One complete family per step; equal family weight by deterministic
        # cyclic schedule. All donor/config views contribute at every step.
        family, caches = grouped['train'][step % len(grouped['train'])]
        metrics = train_step(adapter, mask, optimizer, layers, caches, args.hypothesis,
                             bases=bases, kl_weight=args.kl_weight, sparsity=args.sparsity,
                             control_weight=args.control_weight)
        step += 1
        record = {'step': step, 'family_id': family['id'], **metrics}
        history.append(record); print(json.dumps(record), flush=True)
        with (out/'training_events.jsonl').open('a') as handle:
            handle.write(json.dumps(record)+'\n')
        save_checkpoint(out/'checkpoint.pt', identity, mask, optimizer, step, bases)
    # Save even a zero-step run, then reload to verify the durable round trip.
    save_checkpoint(out/'checkpoint.pt', identity, mask, optimizer, step, bases)
    reloaded_step, _ = load_checkpoint(out/'checkpoint.pt', identity, mask, optimizer)
    assert reloaded_step == step
    evaluations = {s: evaluate(adapter, mask, layers, fs, args.hypothesis, bases) for s, fs in grouped.items()}
    with (out/'metrics.jsonl').open('w') as handle:
        for result in evaluations.values():
            for row in result['rows']:
                handle.write(json.dumps(row)+'\n')
    summaries = {s: {k: v for k, v in r.items() if k != 'rows'} for s, r in evaluations.items()}
    (out/'summary.json').write_text(json.dumps(summaries, indent=2)+'\n')
    (out/'history.json').write_text(json.dumps(history, indent=2)+'\n')
    runtime.update(step=step, max_steps_semantics='total', elapsed_seconds=time.monotonic()-started,
                   checkpoint_reload_verified=True, identity=identity)
    (out/'runtime.json').write_text(json.dumps(runtime, indent=2)+'\n')
    print(json.dumps({'runtime': runtime, 'summary': summaries}))
    return runtime


if __name__ == '__main__':
    main()
