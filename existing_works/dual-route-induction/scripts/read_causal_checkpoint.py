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
    args = ap.parse_args()

    blob = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    meta = blob.get('meta', {})
    corrupt, patched = blob['corrupt'], blob['patched']
    done = blob['done_batches']
    n_examples = int(corrupt['n'])

    n_heads = args.n_heads or meta.get('n_heads')
    assert n_heads, 'n_heads not found in checkpoint meta; pass --n_heads explicitly'

    # same score as end-of-run: mean m1-prob under head-patch minus mean m1-prob under corrupt
    diff = patched['m1_prob'] / patched['n'] - corrupt['m1_prob'] / corrupt['n']  # (n_layers*n_heads,)

    vals, idxs = torch.topk(diff, k=len(diff))
    ranking = [flatidx_to_grididx(int(i), n_heads) for i in idxs]

    scoretype = 'token' if meta.get('random_tok') else 'concept'
    print(f"# {scoretype} copier heads from checkpoint: {done} batches, {n_examples} examples "
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
