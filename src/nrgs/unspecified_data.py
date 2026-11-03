"""Broader-goal kv-transfer data: the kv-(task-)transfer setup one level UP the goal hierarchy, where the next
operation is never written in the prompt.

The instruction names only an algorithm (``Implement merge sort in Python. Name the function `func`.``) and the
prefill is a code prefix natural for both algorithms of a pair, cut at the indentation token before the first line
where they diverge (the pivot)::

    p_src: "Implement A in <lang_src>." + shared prefix  ->  A's next step   (k/v donor)
    p_dst: "Implement B in <lang_dst>." + shared prefix  ->  B's next step   (patched)

The transplanted window is the algorithm NAME (``\\n``-padded to equal length by :func:`FUNC.align_kv_spans`), so it
works across languages. The intervention is the unchanged pivot-onwards kv-transfer of :mod:`src.nrgs.kv_transfer`;
the eval is :func:`src.nrgs.eval.classify_concept` over the WHOLE function (prefill + continuation): which algorithm
does it implement. 12 pairs x 2 directions x 4 language pairs = 96 instances (``tab:kv-goal``). Ported from
``notebooks/diff/05.4_unspecified.ipynb`` on master (same names, prefills and oracle prompts).
"""

import src.nrgs.functional as FUNC
from src.nrgs.eval import classify_concept

INSTRUCTION = "Implement {name} in {lang}. Name the function `func`."
LANG_NAME = {"python": "Python", "cpp": "C++"}
LANGS = {"py-py": ("python", "python"), "py-cpp": ("python", "cpp"), "cpp-py": ("cpp", "python"), "cpp-cpp": ("cpp", "cpp")}
ORACLE_TASK = ("Write a function `func` (in Python or C++) that implements a named algorithm. The function starts "
               "with a prefilled prefix; judge the algorithm the function implements")
VERDICT = {"src": "SRC", "dst": "DST", "both": "both", "neither": "neither", "parse_error": "parse_error"}

PAIRS = {
    'bfs_dfs': dict(
        a=dict(id='bfs', name='breadth-first search graph traversal'),
        b=dict(id='dfs', name='iterative depth-first search graph traversal'),
        prefill=dict(python='<code>\ndef func(graph, start):\n    visited = set()\n   ',
                     cpp='<code>\nstd::vector<int> func(std::vector<std::vector<int>>& graph, int start) {\n    std::vector<bool> visited(graph.size(), false);\n   '),
    ),
    'bsearch_leftmost': dict(
        a=dict(id='binary_search', name='binary search'),
        b=dict(id='leftmost', name='binary search that returns the index of the leftmost occurrence of target'),
        prefill=dict(python='<code>\ndef func(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n       ',
                     cpp='<code>\nint func(std::vector<int>& arr, int target) {\n    int lo = 0, hi = arr.size() - 1;\n    while (lo <= hi) {\n        int mid = lo + (hi - lo) / 2;\n       '),
    ),
    'merge_quick': dict(
        a=dict(id='merge_sort', name='merge sort'),
        b=dict(id='quick_sort', name='quick sort'),
        prefill=dict(python='<code>\ndef func(arr):\n    if len(arr) <= 1:\n        return arr\n   ',
                     cpp='<code>\nstd::vector<int> func(std::vector<int> arr) {\n    if (arr.size() <= 1) {\n        return arr;\n    }\n   '),
    ),
    'dijkstra_bfs': dict(
        a=dict(id='dijkstra', name="Dijkstra's shortest path algorithm from a start node"),
        b=dict(id='bfs_shortest', name='unweighted shortest path from a start node using breadth-first search'),
        prefill=dict(python='<code>\ndef func(graph, start):\n    dist = {start: 0}\n   ',
                     cpp='<code>\nstd::vector<int> func(std::vector<std::vector<int>>& graph, int start) {\n    std::vector<int> dist(graph.size(), INT_MAX);\n    dist[start] = 0;\n   '),
    ),
    'prim_kruskal': dict(
        a=dict(id='prim', name="Prim's minimum spanning tree algorithm"),
        b=dict(id='kruskal', name="Kruskal's minimum spanning tree algorithm"),
        prefill=dict(python='<code>\ndef func(n, edges):\n    mst = []\n   ',
                     cpp='<code>\nint func(int n, std::vector<std::vector<int>>& edges) {\n    int total = 0;\n   '),
    ),
    'bubble_selection': dict(
        a=dict(id='bubble_sort', name='bubble sort'),
        b=dict(id='selection_sort', name='selection sort'),
        prefill=dict(python='<code>\ndef func(arr):\n    n = len(arr)\n    for i in range(n):\n       ',
                     cpp='<code>\nvoid func(std::vector<int>& arr) {\n    int n = arr.size();\n    for (int i = 0; i < n - 1; i++) {\n       '),
    ),
    'fib_rec_iter': dict(
        a=dict(id='fib_recursive', name='the n-th Fibonacci number using naive recursion'),
        b=dict(id='fib_iterative', name='the n-th Fibonacci number iteratively with two running variables'),
        prefill=dict(python='<code>\ndef func(n):\n    if n < 2:\n        return n\n   ',
                     cpp='<code>\nlong long func(int n) {\n    if (n < 2) {\n        return n;\n    }\n   '),
    ),
    'power_fast_naive': dict(
        a=dict(id='fast_power', name='exponentiation by squaring'),
        b=dict(id='naive_power', name='integer power by repeated multiplication in a loop'),
        prefill=dict(python='<code>\ndef func(base, exp):\n    result = 1\n   ',
                     cpp='<code>\nlong long func(long long base, int exp) {\n    long long result = 1;\n   '),
    ),
    'twosum_hash_pointers': dict(
        a=dict(id='twosum_hash', name='two-sum that returns the indices of the two numbers, using a hash map'),
        b=dict(id='twosum_pointers', name='two-sum on a sorted array that returns the indices of the two numbers, using two pointers'),
        prefill=dict(python='<code>\ndef func(nums, target):\n   ',
                     cpp='<code>\nstd::vector<int> func(std::vector<int>& nums, int target) {\n   '),
    ),
    'palindrome_reverse_pointers': dict(
        a=dict(id='pal_reverse', name='a palindrome check that compares the string with its reverse'),
        b=dict(id='pal_pointers', name='a palindrome check that walks two pointers inward from both ends'),
        prefill=dict(python='<code>\ndef func(s):\n   ',
                     cpp='<code>\nbool func(const std::string& s) {\n   '),
    ),
    'toposort_kahn_dfs': dict(
        a=dict(id='kahn', name="topological sort using Kahn's algorithm"),
        b=dict(id='topo_dfs', name='topological sort using depth-first search'),
        prefill=dict(python='<code>\ndef func(graph):\n    result = []\n   ',
                     cpp='<code>\nstd::vector<int> func(std::unordered_map<int, std::vector<int>>& graph) {\n    std::vector<int> result;\n   '),
    ),
    'lcs_dp_memo': dict(
        a=dict(id='lcs_bottom_up', name='longest common subsequence length with bottom-up dynamic programming'),
        b=dict(id='lcs_top_down', name='longest common subsequence length with top-down memoized recursion'),
        prefill=dict(python='<code>\ndef func(a, b):\n    m, n = len(a), len(b)\n   ',
                     cpp='<code>\nint func(const std::string& a, const std::string& b) {\n    int m = a.size(), n = b.size();\n   '),
    ),
}

# display names of the algorithm ids (the paper table)
ALG_TEX = {
    "bfs": "BFS", "dfs": "DFS", "binary_search": "binary search", "leftmost": "leftmost binary search",
    "merge_sort": "merge sort", "quick_sort": "quick sort", "dijkstra": "Dijkstra", "bfs_shortest": "BFS shortest path",
    "prim": "Prim's MST", "kruskal": "Kruskal's MST", "bubble_sort": "bubble sort", "selection_sort": "selection sort",
    "fib_recursive": "recursive Fibonacci", "fib_iterative": "iterative Fibonacci", "fast_power": "fast power",
    "naive_power": "naive power", "twosum_hash": "two-sum (hash map)", "twosum_pointers": "two-sum (two pointers)",
    "pal_reverse": "palindrome (reverse)", "pal_pointers": "palindrome (two pointers)", "kahn": "Kahn's toposort",
    "topo_dfs": "DFS toposort", "lcs_bottom_up": "LCS (bottom-up DP)", "lcs_top_down": "LCS (memoized)",
}


def instances():
    """Every (pair, flip, src_lang, dst_lang) in table order: pair by pair, a->b then b->a, each over LANGS."""
    return [(key, flip, sl, dl) for sl, dl in LANGS.values() for key in PAIRS for flip in (False, True)]


def inst_key(pair_key, flip, src_lang, dst_lang) -> str:
    src, dst = sides(pair_key, flip)
    return f"{pair_key}|{src['id']}->{dst['id']}|{src_lang}-{dst_lang}"


def sides(pair_key, flip=False):
    """(SRC algorithm, DST algorithm) of a pair; ``flip`` swaps them."""
    p = PAIRS[pair_key]
    return (p["b"], p["a"]) if flip else (p["a"], p["b"])


def build_geometry(pair_key, flip, src_lang, dst_lang, *, ctx):
    """Both prompts and the transplanted window (the algorithm name, ``\\n``-padded), as the ``g`` dict
    :func:`src.nrgs.kv_transfer.intervention_stack` takes. The pivot is each prompt's last token."""
    src, dst = sides(pair_key, flip)
    q = lambda alg, lang: INSTRUCTION.format(name=alg["name"], lang=LANG_NAME[lang])
    src_prompt = FUNC.get_prompt(q(src, src_lang), PAIRS[pair_key]["prefill"][src_lang], ctx=ctx)
    dst_prompt = FUNC.get_prompt(q(dst, dst_lang), PAIRS[pair_key]["prefill"][dst_lang], ctx=ctx)
    s0 = FUNC.token_span(src_prompt, src["name"], ctx=ctx)
    d0 = FUNC.token_span(dst_prompt, dst["name"], ctx=ctx)
    src_tok, dst_tok, src_w, dst_w = FUNC.align_kv_spans(ctx.mt, src_prompt, dst_prompt, s0, d0, ctx=ctx)
    return dict(src_tok=src_tok, dst_tok=dst_tok, src_w=src_w, dst_w=dst_w, src_prompt=src_prompt, dst_prompt=dst_prompt)


def function_text(pair_key, gen, lang) -> str:
    """The whole function as written: the prefill (without ``<code>``) + the continuation up to ``</code>``."""
    return PAIRS[pair_key]["prefill"][lang].replace("<code>\n", "", 1) + gen.split("</code>")[0]


def judge_code(pair_key, flip, gen, lang, model_name="claude-sonnet-4-6") -> dict:
    """Which ALGORITHM does the whole function implement: ``verdict`` in SRC / DST / both / neither."""
    src, dst = sides(pair_key, flip)
    r = classify_concept(function_text(pair_key, gen, lang), f"implements {src['name']}", f"implements {dst['name']}",
                         ORACLE_TASK, kind=f"a {LANG_NAME[lang]} function: a prefilled prefix followed by the model's "
                         "continuation. Judge which ALGORITHM the function as a whole implements (BOTH = a mix of the two "
                         "algorithms)", model_name=model_name)
    return dict(r, verdict=VERDICT[r["label"]])


def judge_readout(pair_key, flip, text, model_name="claude-sonnet-4-6") -> dict:
    """Which algorithm does a pivot GoalScope readout name: ``verdict`` in SRC / DST / both / neither."""
    src, dst = sides(pair_key, flip)
    r = classify_concept(text, f"implements {src['name']}", f"implements {dst['name']}", ORACLE_TASK,
                         kind="a short natural-language description of what the model is doing at its current "
                              "position in the code (a verbalized readout of its internal state)",
                         model_name=model_name)
    return dict(r, verdict=VERDICT[r["label"]])
