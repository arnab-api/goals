#!/usr/bin/env python
"""Activation-oracle eval on the multi-predicate ``check_conditions`` setup — two ``--mode``s.

The activation oracle (Karvonen et al. 2025, https://arxiv.org/pdf/2512.15674) is a LoRA adapter
on the *same* base model that verbalizes an injected hidden state: extract h at (layer, token) from
a target-model run, inject it at the oracle's placeholder token, ask a natural-language question.
This script is the activation-oracle counterpart of ``eval_goalscope.py`` — same corpus, same
anchors, same grading rubrics, same record/summary schema — so the two read-out mechanisms are
directly comparable on identical anchors.

  faithfulness  — ask "what are you doing at this line?" about the hidden state at each anchor and
                  grade whether the answer faithfully describes the implementation span.
  causality     — prefill the information-less C++ stub ``bool check(auto value){`` and grade whether
                  the oracle's continuation implements the same *idea* as the Python implementation
                  span (cross-language transfer of the encoded goal).

**Oracle prompt** (``--prompt_style``, default ``goalscope``): the prompt mirrors GoalScope's
destination prompt as closely as the injection mechanism allows — same system turn, the same
instruction text (imported, not copied, from ``goalscope.GOALSCOPE_QUESTION`` /
``eval_goalscope.CAUSAL_INSTRUCTION``), and the same assistant prefill (``""`` / ``CAUSAL_PREFIX``)
continued via ``continue_final_message=True``. The only addition is ``activation_prefix()`` — the
``Layer: N\\n ? \\n`` preamble whose placeholder token receives the injected hidden state. So the two
mechanisms differ *only* in how the goal signal arrives (steered head outputs vs. an injected hidden
state), not in what the model is asked. ``--prompt_style paper`` falls back to Karvonen et al.'s
single user turn (``query_oracle``'s own default).

Caveat on the word "causality" (read this before comparing numbers to ``eval_goalscope.py``):
GoalScope's causality mode intervenes on the **target model** — the goal encoding is steered into
an information-less C++ stub and the *target model's own* generation is redirected. Here, both
modes inject into the **oracle**; the target model's generation is never perturbed. So for the
activation oracle the two modes differ in the *read-out demand* (describe in NL vs. re-implement in
C++), not in whether an intervention happened. Mode names are kept aligned with
``eval_goalscope.py`` on purpose: the corpus, anchors, and grading rubric are shared, so
mode-for-mode numbers answer "does this read-out mechanism surface the goal?" for both.

Pipeline (per sample) — steps 1-3 and 5 are imported verbatim from ``eval_goalscope`` so the two
evals cannot drift apart:

  1. Corpus — sample single-line INT predicates into ``check_conditions(n: int)`` tasks
     (letters naming ``a/b/c``); greedy-decode the body (base weights, oracle adapter disabled).
  2. Oracle spans — rule-based (letters) or ``label_generation`` per-condition impl spans.
  3. Anchors — per assignment line, token positions {prev_end, indent, var, eq, first_impl}.
  4. Read-outs — one forward pass to collect hidden states at ``--extract_layer``, then one oracle
     query per anchor (``src.activation_oracle.query_oracle``).
  5. Scoring — hard eval (grading oracle: is the read-out faithful to the impl_span?) and optional
     k-way (which of the sample's K conditions does the read-out name?).

**Anchor tokenization**: anchors are computed with ``add_special_tokens=False`` to match
``collect_hidden_states`` exactly. This is not cosmetic — Llama-3.3's chat template already emits
``<|begin_of_text|>``, so the default ``add_special_tokens=True`` (what ``prepare_input`` uses) adds
a *second* BOS; reusing ``eval_goalscope.anchors_for_sample`` here would shift every anchor by +1
and silently probe the neighbouring token (verified: the ``var`` anchor would read ``' ='`` instead
of ``' a'``). A per-sample assert pins ``len(offsets) == hidden.shape[0]``.

Known wart inherited on purpose: body generation goes through ``eval_goalscope``'s
``generate_completion`` (``prepare_input`` → double BOS on Llama), while hidden states are collected
single-BOS. So the body is decoded under slightly different conditioning than it is read out under.
Kept as-is because sharing ``generate_completion`` is what makes the corpora match sample-for-sample
against ``eval_goalscope``; the alternative trades that comparability away.

Results save to ``<results>/evaluation/act_oracle/<mode>/<target-model>/<leaf>/`` — config, the
full per-anchor records with raw grading prompts + responses, a per-anchor summary, and per-sample
generations + labels. Parse with ``scripts.evaluation.faithfulness_io`` (``list_act_oracle_runs`` /
``act_oracle_grid_df``). Runs RESUME from the last completed sample by default.

All ``ask_claude`` grading calls use ``use_cache=True``.

Note the two unrelated "oracle"s in play: the **activation oracle** (the LoRA verbalizer under
test) and the **grading oracle** (Claude, scoring read-outs — ``config['grader_model']``).

Usage:
    conda run -n goals python scripts/evaluation/eval_act_oracles.py \
        --mode faithfulness --model meta-llama/Llama-3.3-70B-Instruct --layer_pct 70

    conda run -n goals python scripts/evaluation/eval_act_oracles.py \
        --mode causality --layer_pct 70            # C++ re-implementation transfer

    # layer sweep = one run per layer (the layer is part of the run dir):
    for p in 30 50 70 90; do ... --layer_pct $p; done
"""

import argparse
import hashlib
import json
import logging
import sys
import time

sys.path.append(".")

import torch

from scripts.evaluation import faithfulness_io as fio
from scripts.evaluation.eval_goalscope import (
    ANCHOR_ORDER,
    CAUSAL_HARD_SYS,
    CAUSAL_INSTRUCTION,
    CAUSAL_PREFIX,
    FAITH_HARD_SYS,
    ORACLE_MODEL,
    build_corpus,
    build_prompt,
    compute_summary,
    generate_completion,
    hard_eval,
    kway_assign,
    line_anchors,
    log_summary,
    rule_based_spans,
)
from src.activation_oracle import (
    ORACLE_CONFIGS,
    activation_prefix,
    collect_hidden_states,
    default_oracle_prompt,
    query_oracle,
)
from src.goals.goalscope import GOALSCOPE_QUESTION, GOALSCOPE_SYS
from src.goals.multistep_label import label_generation
from src.models import ModelandTokenizer, get_full_model_path
from src.utils import env_utils, experiment_utils, logging_utils

logger = logging.getLogger(__name__)  # configured in main() via logging_utils

# ══════════════════════════════════════════════════════════════════════════════
# Oracle prompts — mirroring GoalScope's destination prompts
# ══════════════════════════════════════════════════════════════════════════════
# The oracle question and the assistant prefill are taken VERBATIM from the GoalScope destination
# prompts (`src.goals.goalscope._build_goalscope_prompt` / `eval_goalscope.build_causality_prompt`),
# so the only difference between the two read-out mechanisms is *how* the goal signal arrives:
# GoalScope steers head outputs into the prompt, the oracle injects the hidden state at the
# `SPECIAL_TOKEN` placeholder. Everything else — system turn, instruction wording, assistant
# prefill — is identical.
DEFAULT_QUESTIONS = {
    "faithfulness": GOALSCOPE_QUESTION,  # "What are you doing at this line of the code. ..."
    "causality": CAUSAL_INSTRUCTION,  # "You are implementing a simple C++ function ..."
}
# assistant turn the oracle continues; causality prefills the information-less C++ stub
ASSISTANT_PREFILL = {"faithfulness": "", "causality": CAUSAL_PREFIX}


def make_goalscope_style_prompt_builder(mode: str):
    """``prompt_builder`` for ``query_oracle`` that mirrors GoalScope's destination prompt.

    Renders ``[system, user, assistant-prefill]`` with ``continue_final_message=True`` — exactly the
    message list GoalScope templates — with `activation_prefix` (the ``Layer: N\\n ? \\n`` preamble
    carrying the injection placeholder) prepended to the user turn.
    """
    prefill = ASSISTANT_PREFILL[mode]

    def _build(tokenizer, question: str, extract_layer: int) -> str:
        messages = [
            {"role": "system", "content": GOALSCOPE_SYS},
            {"role": "user", "content": activation_prefix(extract_layer) + question},
            {"role": "assistant", "content": prefill},
        ]
        kw = dict(add_generation_prompt=False, continue_final_message=True, tokenize=False)
        try:
            return tokenizer.apply_chat_template(
                messages, chat_template_kwargs={"enable_thinking": False}, **kw
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, **kw)

    return _build


def resolve_prompt_builder(style: str, mode: str):
    """``(builder, readout_prefix)`` for ``--prompt_style``.

    ``goalscope``: mirror the GoalScope probe (system + user + assistant prefill). The read-out is a
    *continuation* of the prefill, so the prefill is prepended before grading — same as
    ``eval_goalscope``'s ``readout_prefix``.
    ``paper``: the paper-faithful single user turn (``query_oracle``'s own default); nothing to prepend.
    """
    if style == "goalscope":
        return make_goalscope_style_prompt_builder(mode), ASSISTANT_PREFILL[mode]
    if style == "paper":
        return None, ""
    raise ValueError(f"unknown --prompt_style {style!r} (want goalscope|paper)")


# ══════════════════════════════════════════════════════════════════════════════
# Oracle adapter
# ══════════════════════════════════════════════════════════════════════════════
def resolve_oracle_lora(model_key: str, override: str | None) -> str:
    if override:
        return override
    if model_key not in ORACLE_CONFIGS:
        raise SystemExit(
            f"no activation oracle registered for {model_key!r}. Known: "
            f"{list(ORACLE_CONFIGS)}. Pass --oracle_lora to use a custom adapter."
        )
    return ORACLE_CONFIGS[model_key]["oracle_lora"]


def load_oracle_adapter(mt, lora_path: str):
    """Load the oracle LoRA onto ``mt._model`` in place and leave it DISABLED.

    The same weights serve as target model (adapters off) and oracle (adapters on inside
    ``query_oracle``), which is what makes this affordable at 70B. Returns
    ``(peft_model, adapter_name, diag)``; ``diag`` records the two adapter-state checks from the
    notebook's smoke test.
    """
    from peft import PeftModel

    adapter = lora_path.replace(".", "_")  # PEFT adapter names must not contain '.'
    if not isinstance(mt._model, PeftModel):
        peft_model = PeftModel.from_pretrained(
            mt._model,
            lora_path,
            adapter_name=adapter,
            is_trainable=False,
            low_cpu_mem_usage=True,
        )
        logger.info(f"oracle LoRA loaded as adapter {adapter!r}")
    else:
        peft_model = mt._model
        if adapter not in peft_model.peft_config:
            peft_model.load_adapter(
                lora_path, adapter_name=adapter, is_trainable=False, low_cpu_mem_usage=True
            )
        logger.info(f"oracle LoRA already present as adapter {adapter!r}")

    # LoraModel's method — does NOT cascade to transformers' PeftAdapterMixin.
    peft_model.base_model.disable_adapter_layers()
    n_lora = sum(
        1 for m in peft_model.modules() if getattr(m, "_disable_adapters", False)
    )
    if n_lora == 0:
        raise SystemExit(
            f"adapter {adapter!r} injected 0 LoRA modules — wrong adapter or model?"
        )

    # PEFT bug guard: set_adapter() alone does not reset _disable_adapters, so without an explicit
    # enable_adapter_layers() every oracle query would silently run the BASE model.
    peft_model.base_model.enable_adapter_layers()
    peft_model.set_adapter(adapter)
    n_active = sum(
        1
        for m in peft_model.modules()
        if hasattr(m, "_disable_adapters") and not m._disable_adapters
    )
    peft_model.base_model.disable_adapter_layers()  # back to base for target-model runs
    if n_active == 0:
        raise SystemExit(
            "oracle adapter never activates after enable+set_adapter — read-outs would be "
            "base-model text. Check the peft version."
        )
    logger.info(
        f"adapter state: {n_lora} LoRA modules (disabled for target runs), "
        f"{n_active} active after enable+set_adapter"
    )
    return peft_model, adapter, dict(n_lora_modules=n_lora, n_active_on_enable=n_active)


def assert_adapters_disabled(peft_model):
    """Target-model work (generation, hidden-state collection) must see base weights only."""
    leaked = [
        n
        for n, m in peft_model.named_modules()
        if hasattr(m, "_disable_adapters") and not m._disable_adapters
    ]
    assert not leaked, (
        f"{len(leaked)} LoRA modules still enabled (e.g. {leaked[:2]}) — target-model "
        f"activations would be contaminated by the oracle adapter"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Anchors — in the oracle's tokenization space (add_special_tokens=False)
# ══════════════════════════════════════════════════════════════════════════════
def anchors_for_sample_ao(mt, full_text, prompt, conditions, impl_spans):
    """``eval_goalscope.anchors_for_sample`` but tokenized with ``add_special_tokens=False``.

    Must match ``collect_hidden_states``' tokenization token-for-token, else anchor indices point
    at the wrong hidden states (Llama-3.3: +1 BOS shift under the default). Returns
    ``(rows, offsets)``; ``offsets`` is returned so the caller can pin it against ``hidden``.
    """
    enc = mt.tokenizer(
        full_text, add_special_tokens=False, return_offsets_mapping=True
    )
    offsets = [tuple(p) for p in enc["offset_mapping"]]
    gen_char_start = len(prompt)
    rows = []
    for k, (cond_text, impl) in enumerate(zip(conditions, impl_spans)):
        var = chr(97 + k)
        rows.append(
            dict(
                cond=k,
                var=var,
                condition=cond_text,
                impl_span=(impl or "").strip(),
                anchors=line_anchors(full_text, offsets, var, gen_char_start),
            )
        )
    return rows, offsets


# ══════════════════════════════════════════════════════════════════════════════
# Driver
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--mode",
        default="faithfulness",
        choices=["faithfulness", "causality"],
        help="faithfulness: NL description vs impl | causality: C++ re-implementation of the idea",
    )
    ap.add_argument(
        "--eval-k-way",
        dest="eval_k_way",
        action="store_true",
        help="also run the k-way grading oracle (which condition does the read-out name?). "
        "Default off. Not implemented for --mode causality (matches eval_goalscope).",
    )
    # ── activation oracle ──
    ap.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct")
    ap.add_argument(
        "--oracle_lora",
        default=None,
        help="HF id / path of the oracle LoRA (default: ORACLE_CONFIGS[model])",
    )
    ap.add_argument(
        "--layer_pct",
        type=int,
        default=70,
        help="extraction layer as %% of depth (70 = the notebook's working setting)",
    )
    ap.add_argument(
        "--extract_layer",
        type=int,
        default=None,
        help="absolute extraction layer; overrides --layer_pct",
    )
    ap.add_argument(
        "--steering_coeff",
        type=float,
        default=1.0,
        help="scales the injected (norm-matched) activation inside the oracle",
    )
    ap.add_argument(
        "--oracle_max_new_tokens",
        type=int,
        default=64,
        help="oracle generation budget per anchor",
    )
    ap.add_argument(
        "--prompt_style",
        default="goalscope",
        choices=["goalscope", "paper"],
        help="oracle prompt shape: 'goalscope' (default) mirrors GoalScope's destination prompt "
        "(system + user + assistant prefill); 'paper' is the single user turn of Karvonen et al.",
    )
    ap.add_argument(
        "--question",
        default=None,
        help="override the oracle question (default: per-mode, see module docstring)",
    )
    ap.add_argument(
        "--question_tag",
        default=None,
        help="short label for the question in the run dir (default: 'default' or an md5 prefix)",
    )
    ap.add_argument("--attn_implementation", default="sdpa", choices=["sdpa", "eager"])
    # ── corpus (identical knobs to eval_goalscope, so corpora match sample-for-sample) ──
    ap.add_argument("--datatype", default="int")
    ap.add_argument("--n_specs", type=int, default=256)
    ap.add_argument("--num_conditions", type=int, nargs="+", default=[3, 4, 5])
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument(
        "--gen_max_new_tokens", type=int, default=256, help="body generation budget"
    )
    ap.add_argument(
        "--span_source",
        default="rule",
        choices=["rule", "oracle"],
        help="impl-span labelling: 'rule' (letters-based, no oracle) | 'oracle' (label_generation)",
    )
    ap.add_argument(
        "--anchors",
        nargs="+",
        default=None,
        choices=ANCHOR_ORDER,
        help=f"restrict to a subset of anchors (default: all of {ANCHOR_ORDER}). "
        f"Each anchor costs one oracle generate() call.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="wipe an existing run dir and redo it; default is to RESUME from the last "
        "completed sample (same args => same run dir)",
    )
    logging_utils.add_logging_args(ap)  # -v (DEBUG) / -q (WARNING); default INFO
    return ap.parse_args()


def main():
    args = parse_args()
    logging_utils.configure(args)  # -v => DEBUG: every read-out + grading verdict logged
    experiment_utils.set_seed(args.seed)
    torch.set_grad_enabled(False)
    results_dir = env_utils.DEFAULT_RESULTS_DIR

    if args.mode == "causality" and args.eval_k_way:
        raise SystemExit("--eval-k-way is not implemented for --mode causality yet")
    hard_sys = CAUSAL_HARD_SYS if args.mode == "causality" else FAITH_HARD_SYS

    oracle_lora = resolve_oracle_lora(args.model, args.oracle_lora)
    prompt_builder, readout_prefix = resolve_prompt_builder(args.prompt_style, args.mode)
    question = args.question or DEFAULT_QUESTIONS[args.mode]
    question_tag = args.question_tag or (
        "default"
        if question == DEFAULT_QUESTIONS[args.mode]
        else hashlib.md5(question.encode()).hexdigest()[:8]
    )
    anchor_names = list(args.anchors) if args.anchors else list(ANCHOR_ORDER)

    # The extraction layer needs mt.n_layer, which needs the model — but the run dir (and the
    # "everything already done" check) must be resolvable first. n_layer comes from the config,
    # so read it off the HF config without materializing weights.
    if args.extract_layer is not None:
        extract_layer, layer_pct = args.extract_layer, None
    else:
        from transformers import AutoConfig

        # resolve exactly like ModelandTokenizer does (local MODEL_DIR first, then HF cache)
        n_layer = AutoConfig.from_pretrained(
            get_full_model_path(args.model), cache_dir=env_utils.HF_CACHE_DIR
        ).num_hidden_layers
        extract_layer, layer_pct = int(n_layer * args.layer_pct / 100), args.layer_pct
        logger.info(
            f"extract layer = {extract_layer} ({args.layer_pct}% of {n_layer} layers)"
        )

    config = dict(
        eval_kind="act_oracle",  # fio.summary_df switches its config columns on this
        mode=args.mode,
        eval_k_way=args.eval_k_way,
        model=args.model,
        oracle_lora=oracle_lora,
        extract_layer=extract_layer,
        layer_pct=layer_pct,
        steering_coeff=args.steering_coeff,
        oracle_max_new_tokens=args.oracle_max_new_tokens,
        prompt_style=args.prompt_style,
        question=question,
        question_tag=question_tag,
        attn_implementation=args.attn_implementation,
        datatype=args.datatype,
        n_specs=args.n_specs,
        num_conditions=list(args.num_conditions),
        seed=args.seed,
        gen_max_new_tokens=args.gen_max_new_tokens,
        span_source=args.span_source,
        anchors=anchor_names,
        grader_model=ORACLE_MODEL,
        # with the goalscope-style prompt the read-out continues an assistant prefill, so the
        # prefill is prepended before grading — exactly like eval_goalscope's readout_prefix
        readout_prefix=readout_prefix,
        hard_rubric_tag=hashlib.md5(hard_sys.encode()).hexdigest()[:8],
    )
    run_dir = fio.act_oracle_run_dir_for(config, results_dir)

    # ── overwrite vs resume ──
    if args.overwrite and run_dir.exists():
        for f in ("records.jsonl", "samples.jsonl", "samples.json", "summary.json"):
            (run_dir / f).unlink(missing_ok=True)
        logger.info(f"--overwrite: cleared existing run at {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"run dir -> {run_dir}")

    # corpus is deterministic from (n_specs, num_conditions, seed, datatype) and needs no model,
    # so resolve what's already done before paying for the 70B load.
    samples = build_corpus(args.n_specs, args.num_conditions, args.seed, args.datatype)
    completed = fio.read_completed_spec_ids(run_dir)
    kept = fio.prune_orphan_records(run_dir, completed)
    remaining = [s for s in samples if s.spec.spec_id not in set(completed)]
    logger.info(
        f"corpus: {len(samples)} samples | already done: {len(completed)} "
        f"({kept} records kept) | remaining: {len(remaining)}"
    )

    if not remaining:
        recs = (
            [
                json.loads(l)
                for l in fio.records_path(run_dir).read_text().splitlines()
                if l.strip()
            ]
            if fio.records_path(run_dir).exists()
            else []
        )
        summary = compute_summary(recs, n_samples=len(completed))
        fio.write_summary(run_dir, summary)
        logger.info(f"all samples already complete; summary refreshed -> {run_dir}")
        log_summary(summary, args.mode)
        return

    mt = ModelandTokenizer(
        model_key=args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=args.attn_implementation,
    )
    if args.extract_layer is None:  # pin the config-derived layer against the real model
        assert extract_layer == int(mt.n_layer * args.layer_pct / 100), (
            f"extract layer {extract_layer} disagrees with the loaded model "
            f"({mt.n_layer} layers) — HF config and weights out of sync?"
        )
    assert 0 <= extract_layer < mt.n_layer, (
        f"--extract_layer {extract_layer} out of range for {mt.n_layer} layers"
    )

    peft_model, adapter, adapter_diag = load_oracle_adapter(mt, oracle_lora)
    config["oracle_adapter"] = adapter
    config["adapter_diag"] = adapter_diag
    config["n_layer"] = mt.n_layer
    config["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    fio.write_config(run_dir, config)  # self-describing even while in progress
    logger.info(
        f"mode={args.mode} | layer={extract_layer}/{mt.n_layer} | coeff={args.steering_coeff} "
        f"| anchors={anchor_names} | question={question!r}"
    )

    oracle_kwargs = dict(
        peft_model=peft_model,
        tokenizer=mt.tokenizer,
        question=question,
        oracle_adapter_name=adapter,
        extract_layer=extract_layer,
        max_new_tokens=args.oracle_max_new_tokens,
        steering_coeff=args.steering_coeff,
        prompt_builder=prompt_builder,  # None => query_oracle's paper-faithful default
    )
    # log the exact oracle prompt once — the injection site and the prefill are easy to get wrong
    _example_prompt = (
        prompt_builder(tokenizer=mt.tokenizer, question=question, extract_layer=extract_layer)
        if prompt_builder is not None
        else default_oracle_prompt(mt.tokenizer, question, extract_layer)
    )
    logger.info(f"oracle prompt [{args.prompt_style}]:\n{_example_prompt}")
    config["oracle_prompt"] = _example_prompt
    fio.write_config(run_dir, config)

    rec_fh = open(fio.records_path(run_dir), "a")
    samp_fh = open(fio.samples_path(run_dir), "a")
    n_done = len(completed)
    smoke_done = False
    for sample in remaining:
        sample_records = []
        assert_adapters_disabled(peft_model)  # generation must use base weights
        prompt = build_prompt(mt, sample)
        completion = generate_completion(mt, prompt, args.gen_max_new_tokens)
        full_text = prompt + completion

        # impl-span + condition text, either rule-based (letters, no oracle) or oracle-labelled
        if args.span_source == "rule":
            conditions = list(sample.spec.spec_texts)  # ground-truth descriptions
            impl_spans = rule_based_spans(
                full_text, len(prompt), len(sample.spec.predicate_ids)
            )
            labels_meta = [
                dict(step_text=c, gen_code=(s or ""), gen_token_range=None)
                for c, s in zip(conditions, impl_spans)
            ]
        else:  # oracle
            labeled = label_generation(mt, sample, prompt, completion, use_cache=True)
            conditions = [sl.step_text for sl in labeled.labels]
            impl_spans = [sl.gen_code.strip() for sl in labeled.labels]
            labels_meta = [
                dict(
                    step_text=sl.step_text,
                    gen_code=sl.gen_code,
                    gen_token_range=list(sl.gen_token_range)
                    if sl.gen_token_range
                    else None,
                )
                for sl in labeled.labels
            ]

        rows, offsets = anchors_for_sample_ao(
            mt, full_text, prompt, conditions, impl_spans
        )
        conds = [r["condition"] for r in rows]
        if not any(r["anchors"] for r in rows):
            logger.warning(f"[{sample.spec.spec_id}] no anchors found — skipping")
            continue

        # one forward pass per sample: hidden states for every token at the extraction layer
        hidden, _ = collect_hidden_states(mt, full_text, layer=extract_layer)
        assert len(offsets) == hidden.shape[0], (
            f"tokenization drift: {len(offsets)} anchor offsets vs {hidden.shape[0]} hidden "
            f"states — anchor indices would not address the right tokens"
        )

        for row in rows:
            true_id = sample.spec.predicate_ids[row["cond"]]
            for aname in anchor_names:
                pos = row["anchors"].get(aname)
                if pos is None:
                    continue
                # readout_prefix restores the assistant prefill the oracle was continuing, so the
                # grader sees a complete-ish snippet (mirrors eval_goalscope)
                ro = readout_prefix + query_oracle(
                    activation_vec=hidden[pos], **oracle_kwargs
                )
                hv = hard_eval(ro, row["impl_span"], args.mode)
                logger.debug(
                    "[%s c%d %s pos=%d] impl=%r cond=%r",
                    sample.spec.spec_id,
                    row["cond"],
                    aname,
                    pos,
                    row["impl_span"],
                    row["condition"],
                )
                logger.debug("    read-out : %r", ro)
                logger.debug(
                    "    HARD  verdict=%s  grader_response=%r", hv["verdict"], hv["response"]
                )
                rec = dict(
                    spec=sample.spec.spec_id,
                    cond=row["cond"],
                    var=row["var"],
                    anchor=aname,
                    pos=pos,
                    extract_layer=extract_layer,
                    readout=ro,
                    impl_span=row["impl_span"],
                    condition=row["condition"],
                    true_pred_id=true_id,
                    hard=hv["verdict"],
                    hard_prompt=hv["prompt"],
                    hard_response=hv["response"],
                )
                if args.eval_k_way:
                    kv = kway_assign(ro, conds)
                    logger.debug(
                        "    KWAY  pred=%s (true %d)  grader_response=%r",
                        kv["pred"],
                        row["cond"],
                        kv["response"],
                    )
                    rec.update(
                        kway_pred=kv["pred"],
                        kway_true=row["cond"],
                        kway_prompt=kv["prompt"],
                        kway_response=kv["response"],
                    )
                sample_records.append(rec)

        # startup sanity check (notebook cell 8): distinct hidden states must give distinct
        # read-outs, else the injection silently isn't reaching the oracle.
        if not smoke_done and len(sample_records) >= 2:
            a, b = sample_records[0]["readout"], sample_records[-1]["readout"]
            if a.strip() == b.strip():
                logger.warning(
                    "SMOKE: read-outs at pos %d and %d are IDENTICAL (%r) — injection may not be "
                    "reaching the oracle; verify the adapter and placeholder token",
                    sample_records[0]["pos"],
                    sample_records[-1]["pos"],
                    a[:80],
                )
            else:
                logger.info("SMOKE: distinct positions give distinct read-outs (injection live)")
            smoke_done = True

        sample_meta = dict(
            spec_id=sample.spec.spec_id,
            predicate_ids=list(sample.spec.predicate_ids),
            prompt=prompt,
            completion=completion,
            labels=labels_meta,
            anchors=[
                dict(cond=r["cond"], var=r["var"], anchors=r["anchors"]) for r in rows
            ],
        )
        # records FIRST, then the sample meta line: a sample counts as "done" only once its meta
        # lands, so a crash mid-sample leaves prunable orphan records, not a phantom-complete one.
        for r in sample_records:
            fio.append_jsonl(rec_fh, r)
        fio.append_jsonl(samp_fh, sample_meta)
        n_done += 1
        logger.info(
            f"[{n_done}/{len(samples)}] {sample.spec.spec_id}: "
            f"{len(sample_records)} anchor read-outs scored"
        )

    rec_fh.close()
    samp_fh.close()

    # ── aggregate from everything on disk (resumed rows included) ──
    all_records = [
        json.loads(l)
        for l in fio.records_path(run_dir).read_text().splitlines()
        if l.strip()
    ]
    n_samples = len(fio.read_completed_spec_ids(run_dir))
    summary = compute_summary(all_records, n_samples)
    fio.write_summary(run_dir, summary)
    logger.info(f"saved -> {run_dir}")
    log_summary(summary, args.mode)


if __name__ == "__main__":
    main()
