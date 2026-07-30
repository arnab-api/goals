'''
Dump the (noisy, partial-run) copier-head ranking from a causal_scores checkpoint.

The checkpoint stores running-sum accumulators, so at any completed batch the copying score
    diff = patched.m1_prob / n  -  corrupt.m1_prob / n
is a valid estimate of each head's copying score -- just over fewer examples (noisier), and
early on skewed toward whichever n-gram bucket has run so far (buckets are processed in order).

Usage:
  python read_causal_checkpoint.py <path/to/{concept,token}_len30_n1024.ckpt.pt> \
      [--topk 30] [--n_heads N] [--out ranking.json]
'''
import json
import os
import argparse
import torch

from utils import flatidx_to_grididx, json_tuple_keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt', help='path to a *.ckpt.pt written by causal_scores.py')
    ap.add_argument('--topk', type=int, default=30)
    ap.add_argument('--n_heads', type=int, default=None,
                    help='heads per layer; default = read from checkpoint meta')
    ap.add_argument('--out', default=None, help='optional json to write full ranking + scores')
    ap.add_argument('--score', default=None, choices=['m1_prob', 'm2_prob'],
                    help="which accumulator to difference; default m1 for concept, m2 for token")
    args = ap.parse_args()

    blob = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    meta = blob.get('meta', {})
    corrupt, patched = blob['corrupt'], blob['patched']
    done = blob['done_batches']
    n_examples = int(corrupt['n'])

    scoretype_hint = ('token' if meta.get('random_tok') or 'token' in os.path.basename(args.ckpt)
                      else 'concept')
    n_heads = args.n_heads or meta.get('n_heads')
    assert n_heads, 'n_heads not found in checkpoint meta; pass --n_heads explicitly'

    # concept vs token are NOT the same quantity (see convert_causal_scores.py, which writes the
    # canonical head_orderings):  concept = m1 diff (real entities), token = m2 diff (random toks).
    field = args.score or ('m2_prob' if scoretype_hint == 'token' else 'm1_prob')
    diff = patched[field] / patched['n'] - corrupt[field] / corrupt['n']  # (n_layers*n_heads,)

    vals, idxs = torch.topk(diff, k=len(diff))
    ranking = [flatidx_to_grididx(int(i), n_heads) for i in idxs]

    scoretype = scoretype_hint
    print(f"# {scoretype} copier heads [{field}] from checkpoint: {done} batches, {n_examples} examples "
          f"(noisy / possibly length-skewed if early) -- top {args.topk}")
    for r in range(min(args.topk, len(ranking))):
        layer, head = ranking[r]
        print(f"  {r + 1:3d}. layer {layer:3d} head {head:3d}   score={vals[r].item():+.6f}")

    if args.out:
        score_map = {flatidx_to_grididx(i, n_heads): diff[i].item() for i in range(len(diff))}
        with open(args.out, 'w') as f:
            json.dump({'meta': meta, 'done_batches': done, 'n_examples': n_examples,
                       'ranking': ranking, 'scores': json_tuple_keys(score_map)}, f)
        print(f"wrote {args.out}")


if __name__ == '__main__':
    main()
