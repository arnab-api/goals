'''
Patch outputs of individual heads across examples at the -2 position, 
and save how much each head:
    - promotes `Ed` at the -2 position (logit diff, prob. diff)
    - promotes `mont` at the -1 position (logit diff, prob. diff)
    
Clean: foo bar [qux] Ed.mont.on\nfoo bar [qux] Ed
Corr:  elf min Apple MININGquee\nfoo bar [qux] Ed

You can also replace CounterFact entities with random tokens using `--random_tok_entities`
Then, run `convert_causal_scores.py` to convert the resulting .pkl files into readable json scores. 
'''
import os 
import json 
import pickle 
import argparse 
import random 
import torch 
import numpy as np 
import pandas as pd 

from random import shuffle
from tqdm import tqdm 
from collections import defaultdict
from nnsight import LanguageModel
from datasets import load_dataset

from utils import pile_chunk, json_tuple_keys, flatidx_to_grididx

# HF cache root (Qwen weights + tokenizer live under /disk/u/models on this machine).
# Override with the HF_CACHE env var (e.g. to point at ~/.cache/huggingface for the paper models).
HF_CACHE = os.environ.get('HF_CACHE', '/disk/u/models')

import time
_T0 = time.time()
def log(msg):
    """Flushing, timestamped progress line (flush=True so it shows through `tee`/redirection)."""
    el = time.time() - _T0
    print(f"[+{int(el)//60:02d}:{int(el)%60:02d}] {msg}", flush=True)

def flat_to_dict(flattensor, n_heads=32):
    d = {}
    for idx in range(len(flattensor)):
        d[flatidx_to_grididx(idx, n_heads)] = flattensor[idx].item()
    return d 

def flat_to_ranking(flattensor, n_heads=32):
    _, idxs = torch.topk(flattensor, k=len(flattensor))
    return [flatidx_to_grididx(i, n_heads) for i in idxs]

class ChunkOutputSaver:
    def __init__(self, name, n_heads):
        self.name = name 
        self.n_heads = n_heads
        self.n = 0 
        self.correct = torch.zeros(n_heads) # specifically it's whether m1 is correct 
        self.m1_prob = torch.zeros(n_heads)
        self.m2_prob = torch.zeros(n_heads)
        self.m1_logit = torch.zeros(n_heads)
        self.m2_logit = torch.zeros(n_heads)
    
    def __repr__(self):
        return f'ChunkOutputSaver-{self.name}-heads{self.n_heads}'

    def update(self, correct, m1_prob, m2_prob, m1_logit, m2_logit):
        '''
        Each input array should be shape (bsz, n_heads)
        '''
        assert len(correct) == len(m2_prob)
        assert len(m2_prob) == len(m1_prob)
        self.n += len(correct)
        self.correct += correct.sum(dim=0)
        self.m1_prob += m1_prob.sum(dim=0)
        self.m2_prob += m2_prob.sum(dim=0)
        self.m1_logit += m1_logit.sum(dim=0)
        self.m2_logit += m2_logit.sum(dim=0)
    
    def get_acc(self):
        return self.correct / self.n 

    def get_m1(self):
        return self.m1_prob / self.n 

    def get_m2(self):
        return self.m2_prob / self.n 

    def get_m1_logit(self):
        return self.m1_logit / self.n 
    
    def get_m2_logit(self):
        return self.m2_logit / self.n

def _saver_state(s):
    """Plain-tensor snapshot of a ChunkOutputSaver (no custom class in the pickle)."""
    return {'name': s.name, 'n_heads': s.n_heads, 'n': s.n,
            'correct': s.correct, 'm1_prob': s.m1_prob, 'm2_prob': s.m2_prob,
            'm1_logit': s.m1_logit, 'm2_logit': s.m2_logit}

def _load_saver(state):
    s = ChunkOutputSaver(state['name'], state['n_heads'])
    s.n = state['n']
    s.correct = state['correct']; s.m1_prob = state['m1_prob']; s.m2_prob = state['m2_prob']
    s.m1_logit = state['m1_logit']; s.m2_logit = state['m2_logit']
    return s

def save_checkpoint(ckpt_file, done_batches, clean_results, corrupt_results, patched_results, meta):
    """Atomically dump the running accumulators so a crashed run can resume by batch index."""
    tmp = f"{ckpt_file}.{os.getpid()}.tmp"
    torch.save({'done_batches': done_batches,
                'clean': _saver_state(clean_results),
                'corrupt': _saver_state(corrupt_results),
                'patched': _saver_state(patched_results),
                'meta': meta}, tmp)
    os.replace(tmp, ckpt_file)

# generate clean and corrupted prompts
def generate_seq_batch(entities, pile, tok):
    # derive the sequence-start + newline tokens from the tokenizer (model-generic).
    # llama: reproduces the original [1] (BOS) / [13] ('\n'); Qwen2.5: [<|endoftext|>] / [198].
    start = tok('', bos=True)            # list, e.g. [1] (llama) or [<|endoftext|>] (Qwen)
    newline = tok('\n', bos=False)[-1]   # 13 (llama) / 198 (Qwen)

    clean, corrupt = [], []
    for ent in entities:
        rand = pile_chunk(args.sequence_len - len(ent), pile, tok, shuf_pile=True)
        ent_chunk_full = rand + ent
        ent_chunk_trunc = rand + [ent[0]]
        clean_prompt = start + ent_chunk_full + [newline] + ent_chunk_trunc

        rand2 = pile_chunk(args.sequence_len, pile, tok, shuf_pile=True)
        corrupt_prompt = start + rand2 + [newline] + ent_chunk_trunc
        # print(repr(tokenizer.decode(clean_prompt)))
        # print(repr(tokenizer.decode(corrupt_prompt)))
        clean.append(clean_prompt)
        corrupt.append(corrupt_prompt)
    return clean, corrupt

def inference_logits(model, sequences):
    remote = model.config._name_or_path == 'meta-llama/Meta-Llama-3.1-70B'
    with torch.no_grad():
        with model.trace(sequences, remote=remote):
            logits = model.output.logits.save()
    return logits.detach().cpu()

def stats_from_logits(logits, entities):
    # trying to predict the second token of the entity 
    correct = logits[:, -1].argmax(dim=-1) == entities[:, 1]

    probs = logits.softmax(dim=-1)
    m1_probs = probs[torch.arange(len(entities)), -1, entities[:, 1]]
    m2_probs = probs[torch.arange(len(entities)), -2, entities[:, 0]]

    m1_logits = logits[torch.arange(len(entities)), -1, entities[:, 1]]
    m2_logits = logits[torch.arange(len(entities)), -2, entities[:, 0]]

    return correct, m1_probs, m2_probs, m1_logits, m2_logits

def no_patching(model, sequences, entities):
    logits = inference_logits(model, sequences) 
    return stats_from_logits(logits, entities)

def get_head_activations(model, prompt, layer):
    remote = model.config._name_or_path == 'meta-llama/Meta-Llama-3.1-70B'
    with torch.no_grad():
        if model.config._name_or_path == 'EleutherAI/pythia-6.9b':
            with model.trace(prompt, remote=remote):
                o_proj_inp = model.gpt_neox.layers[layer].attention.dense.inputs[0][0].save()
        else: 
            with model.trace(prompt, remote=remote):
                o_proj_inp = model.model.layers[layer].self_attn.o_proj.inputs[0][0].save()
                
        # [bsz, seq_len, model_dim] -> [bsz, seq_len, n_heads, head_dim]
        heads_per_layer = model.config.num_attention_heads 
        head_dim = model.config.hidden_size // heads_per_layer 
        return o_proj_inp.view(*o_proj_inp.shape[:-1], heads_per_layer, head_dim)

# patches the -2 index for a batch of sequences
def patch_head_m2(model, clean_seq, corr_seq, entities, desc="", log_every=8):
    heads_per_layer = model.config.num_attention_heads
    head_dim = model.config.hidden_size //  heads_per_layer
    n_layers = model.config.num_hidden_layers

    # will be 1024-dimensional lists
    correct = []
    m1_probs = []
    m2_probs = []
    m1_logits = []
    m2_logits = []
    for layer in range(n_layers):
        # (bsz, seq_len, n_heads, head_dim)
        clean_heads = get_head_activations(model, clean_seq, layer)
        if (layer + 1) % log_every == 0 or layer == n_layers - 1:
            log(f"    {desc} patched layer {layer + 1}/{n_layers} ({heads_per_layer} heads/layer)")
        
        for head_idx in range(heads_per_layer):
            remote = model.config._name_or_path == 'meta-llama/Meta-Llama-3.1-70B'
            with torch.no_grad():
                with model.trace(corr_seq, remote=remote): 
                    if model.config._name_or_path == 'EleutherAI/pythia-6.9b': 
                        tup = model.gpt_neox.layers[layer].attention.dense.inputs
                    else: 
                        tup = model.model.layers[layer].self_attn.o_proj.inputs

                    original_shape = tup[0][0].shape # [bsz, seq_len, model_dim]
                    sub = tup[0][0].view(original_shape[0], original_shape[1], heads_per_layer, head_dim)

                    sub[:, -2, head_idx, :] = clean_heads[:, -2, head_idx, :]
                    sub = sub.view(original_shape)
                    new_tup = ((sub,), tup[1])

                    if model.config._name_or_path == 'EleutherAI/pythia-6.9b':
                        model.gpt_neox.layers[layer].attention.dense.inputs = new_tup
                    else: 
                        model.model.layers[layer].self_attn.o_proj.inputs = new_tup

                    logits = model.output.logits.save()
            
            # [bsz] bits that we want to collect 
            c, m1, m2, m1l, m2l = stats_from_logits(logits.detach().cpu(), entities)
            correct.append(c); m1_probs.append(m1); m2_probs.append(m2)
            m1_logits.append(m1l); m2_logits.append(m2l)
    
    # wait i want (bsz, 1024) tensors 
    return torch.stack(correct).T, torch.stack(m1_probs).T, torch.stack(m2_probs).T, \
        torch.stack(m1_logits).T, torch.stack(m2_logits).T

        
def main(args):
    random.seed(8)
    torch.manual_seed(8)
    np.random.seed(8)
    model_name = args.model.split('/')[-1]
    assert args.bsz <= args.n // 4 

    remote = args.model == 'meta-llama/Meta-Llama-3.1-70B'

    log(f"loading model {args.model} (HF_CACHE={HF_CACHE}, random_tok={args.random_tok_entities}) — sharding across visible GPUs, may take minutes ...")
    if args.ckpt is not None:
        assert args.model in ['allenai/OLMo-2-1124-7B', 'EleutherAI/pythia-6.9b']
        model = LanguageModel(args.model, device_map='auto', dispatch=(not remote), revision=args.ckpt)
    else:
        model = LanguageModel(args.model, device_map='auto', dispatch=(not remote), cache_dir=HF_CACHE)
    tokenizer = model.tokenizer
    log(f"model loaded: {model.config.num_hidden_layers} layers x {model.config.num_attention_heads} heads "
        f"= {model.config.num_hidden_layers * model.config.num_attention_heads} traces/batch")

    # device-placement check: CPU/disk offload makes each forward ~100x slower -> flag it loudly.
    try:
        dm = (getattr(getattr(model, '_model', model), 'hf_device_map', None)
              or getattr(model, 'hf_device_map', None))
        if dm:
            counts = {}
            for v in dm.values():
                counts[str(v)] = counts.get(str(v), 0) + 1
            log(f"device map: {counts}")
            if any(str(v) in ('cpu', 'disk') for v in dm.values()):
                log("*** WARNING: part of the model is offloaded to CPU/disk (device_map='auto'). "
                    "Per-batch time will be HUGE. Give this job MORE GPU memory (more GPUs in the "
                    "baulab worker slot) so the whole model fits on GPU. ***")
    except Exception as e:
        log(f"(could not read device map: {e})")

    # tokenization function for any model 
    def tok(s, bos=False, model=model):
        if 'llama' in model.config._name_or_path:
            if not bos: 
                return model.tokenizer(s)['input_ids'][1:]
            else:
                return model.tokenizer(s)['input_ids']
        elif 'OLMo' in model.config._name_or_path or 'pythia' in model.config._name_or_path:
            if not bos:
                return model.tokenizer(s)['input_ids']
            else:
                return [model.tokenizer.bos_token_id] + model.tokenizer(s)['input_ids']
        elif 'Qwen' in model.config._name_or_path:
            # Qwen2.5 adds no special tokens by default and has no BOS;
            # use <|endoftext|> (the pretraining doc separator) as the sequence-start token.
            ids = model.tokenizer(s)['input_ids']
            if not bos:
                return ids
            else:
                return [model.tokenizer.convert_tokens_to_ids('<|endoftext|>')] + ids

    # load in pile sample to use as basic material that we shuffle around
    log("loading pile-10k ...")
    pile = load_dataset('NeelNanda/pile-10k')['train']
    log("pile-10k loaded; building + sorting entities by token length ...")

    # dummy entities for comparison 
    sorted_entities = defaultdict(list)
    if args.random_tok_entities:
        for i in range(args.n):
            doc_toks = []
            while len(doc_toks) < 5:
                doc = pile.shuffle()[0]['text']
                doc_toks = tok(doc)

            shuffle(doc_toks)
            if i % 4 == 0: 
                sorted_entities['bigram'].append(doc_toks[:2])
            elif i % 4 == 1: 
                sorted_entities['trigram'].append(doc_toks[:3])
            elif i % 4 == 2:
                sorted_entities['fourgram'].append(doc_toks[:4])
            elif i % 4 == 3: 
                sorted_entities['fivegram'].append(doc_toks[:5])

    # load and sort entities of different token lengths
    else:
        str_entities = list(pd.read_csv('../data/counterfact_expanded.csv')['subject'])
        for ent in str_entities:
            toks = tok(ent)
            if len(toks) == 2:
                sorted_entities['bigram'].append(toks)
            elif len(toks) == 3: 
                sorted_entities['trigram'].append(toks)
            elif len(toks) == 4: 
                sorted_entities['fourgram'].append(toks)
            elif len(toks) == 5: 
                sorted_entities['fivegram'].append(toks)

    # accumulators. patching experiments will log across each attention head
    n_heads = model.config.num_attention_heads * model.config.num_hidden_layers
    clean_results = ChunkOutputSaver('clean', 1)
    corrupt_results = ChunkOutputSaver('corrupt', 1)
    patched_results = ChunkOutputSaver('patched', n_heads)

    total_batches = sum(len(ents[: args.n // 4]) // args.bsz for ents in sorted_entities.values())
    log("entities/bucket: " + ", ".join(f"{k}={len(v)}" for k, v in sorted_entities.items())
        + f"  ->  {total_batches} batches total (bsz={args.bsz})")

    # --- checkpointing setup (resume across crashes; per-batch, atomic) ---
    # checkpoint is namespaced by scoretype so the concurrent concept/token jobs never clobber
    # each other (they share model_name + fname). meta must match to resume; else pass --overwrite.
    scoretype = 'token' if args.random_tok_entities else 'concept'
    fname = f'len{args.sequence_len}_n{args.n}'
    ckpt_dir = f'../cache/causal_scores/{model_name}/' + (f'{args.ckpt}/' if args.ckpt is not None else '')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_file = os.path.join(ckpt_dir, f'{scoretype}_{fname}.ckpt.pt')
    meta = {'model': args.model, 'n': args.n, 'bsz': args.bsz,
            'sequence_len': args.sequence_len, 'random_tok': args.random_tok_entities, 'ckpt': args.ckpt,
            'n_heads': model.config.num_attention_heads, 'n_layers': model.config.num_hidden_layers}

    resume_from = 0
    if os.path.exists(ckpt_file) and not args.overwrite:
        blob = torch.load(ckpt_file, map_location='cpu', weights_only=False)
        if blob.get('meta') != meta:
            raise SystemExit("[checkpoint] config mismatch with existing checkpoint; pass --overwrite to discard.\n"
                             f"  existing: {blob.get('meta')}\n  current:  {meta}")
        resume_from = blob['done_batches']
        clean_results   = _load_saver(blob['clean'])
        corrupt_results = _load_saver(blob['corrupt'])
        patched_results = _load_saver(blob['patched'])
        log(f"resuming from checkpoint {ckpt_file}: {resume_from}/{total_batches} batches already done")
    elif os.path.exists(ckpt_file):
        log(f"--overwrite set: ignoring existing checkpoint {ckpt_file}")

    # each batch only has one entity length because of bsz defined above
    done_batches = 0
    for l, ents in sorted_entities.items():
        selected_ents = ents[ : args.n // 4]
        n_batches = len(selected_ents) // args.bsz
        log(f"=== bucket '{l}': {len(selected_ents)} ents -> {n_batches} batches "
            f"(e.g. {tokenizer.decode(selected_ents[0])!r})")

        for batch_idx in range(n_batches):
            if done_batches < resume_from:
                # already accounted for in the restored accumulators — fast-forward past it
                done_batches += 1
                continue
            t_b = time.time()
            batch_ents = selected_ents[batch_idx * args.bsz : (batch_idx + 1) * args.bsz]
            batch_clean, batch_corr = generate_seq_batch(batch_ents, pile, tok)

            batch_ents = torch.tensor(batch_ents)
            clean_results.update(*no_patching(model, batch_clean, batch_ents))
            corrupt_results.update(*no_patching(model, batch_corr, batch_ents))

            # patch outputs of each head from [-2] index
            patched_results.update(
                *patch_head_m2(model, batch_clean, batch_corr, batch_ents,
                               desc=f"[{l} b{batch_idx + 1}/{n_batches}]")
            )
            done_batches += 1
            # atomically checkpoint the running accumulators after every completed batch
            save_checkpoint(ckpt_file, done_batches, clean_results, corrupt_results,
                            patched_results, meta)
            log(f"  bucket '{l}' batch {batch_idx + 1}/{n_batches} done in {time.time() - t_b:.0f}s "
                f"| overall {done_batches}/{total_batches} [checkpointed]")

    all_results = [clean_results, corrupt_results, patched_results]
    log("all batches done; writing results ...")

    path = f'../cache/causal_scores/{model_name}/'
    path += f'{args.ckpt}/' if args.ckpt is not None else ''
    os.makedirs(path, exist_ok=True)

    fname = f'len{args.sequence_len}_n{args.n}'

    # save all results just in case
    log(f"saving pkl -> {path + fname + '.pkl'}")
    with open(path + fname + '.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    
    # save important results as json 
    # convert to dictionaries with (layer, head_idx) tuples
    scoretype = 'token' if args.random_tok_entities else 'concept'

    diff = patched_results.get_m1() - corrupt_results.get_m1()
    copying_scores = flat_to_dict(diff, n_heads=model.config.num_attention_heads)
    with open(path + f'{scoretype}_copying_{fname}.json', 'w') as f:
        json.dump(json_tuple_keys(copying_scores), f)

    # save head rankings  
    rank_path = f'../cache/head_orderings/{model_name}/'
    rank_path += f'{args.ckpt}/' if args.ckpt is not None else ''
    os.makedirs(rank_path, exist_ok=True)

    copying_rankings = flat_to_ranking(diff, n_heads=model.config.num_attention_heads)
    json_name = f'{scoretype}_copying'
    json_name += f'_{fname}.json' if (args.n != 1024 or args.sequence_len != 30) else '.json'
    with open(rank_path + json_name, 'w') as f:
        json.dump(copying_rankings, f)

    log(f"DONE ({scoretype}): scores -> {path}{scoretype}_copying_{fname}.json | ranking -> {rank_path}{json_name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='meta-llama/Llama-2-7b-hf', 
                        choices=[
                            'meta-llama/Llama-2-7b-hf',
                            'meta-llama/Meta-Llama-3-8B',
                            'allenai/OLMo-2-1124-7B',
                            'EleutherAI/pythia-6.9b',
                            'meta-llama/Meta-Llama-3.1-70B',
                            'meta-llama/Llama-3.2-3B',
                            'allenai/OLMo-2-0425-1B',
                            'Qwen/Qwen2.5-Coder-7B-Instruct',
                            'Qwen/Qwen2.5-Coder-14B-Instruct',
                            'Qwen/Qwen2.5-Coder-32B-Instruct',
                            ])
    parser.add_argument('--ckpt', default=None, type=str)
    parser.add_argument('--n', default=1024, type=int)
    parser.add_argument('--bsz', default=32, type=int)
    parser.add_argument('--sequence_len', default=30, type=int)
    parser.add_argument('--random_tok_entities', action='store_true')
    parser.set_defaults(random_tok_entities=False)
    parser.add_argument('--overwrite', action='store_true',
                        help='ignore any existing checkpoint and recompute from scratch')
    parser.set_defaults(overwrite=False)
    args = parser.parse_args()
    main(args)