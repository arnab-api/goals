"""GoalScope: verbalize the goal heads' encoding by steering it into a read-out probe.

This module provides:
- Greedy generation helpers (plain no-cache, head-replacement patched, additive-norm steered; batched)
- The GoalScope destination prompts (NL faithfulness probe, C++ causality stub)
- ``apply_goalscope`` (goal-head read-out per token) and ``OVReader`` (OV-write residual probe)
"""

from __future__ import annotations

import warnings
from typing import Callable, Collection, Literal, Optional, Union

import torch
from tqdm.auto import tqdm

from src.functional import _LOOP_CUT_SHORT, PatchSpec, generate_with_patch
from src.goals.head_loc_attn import (
    HeadMask,
    collect_head_steering_vecs,
    get_o_proj_inputs_batch,
)
from src.tokens import chat_prompt, find_token_range, prepare_input
from src.utils.typing import TokenizerOutput

# ---------------------------------------------------------------------------
# Greedy generation helpers
# ---------------------------------------------------------------------------


def greedy_continue(
    mt,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    stop_ids: Optional[Collection[int]] = None,
    interventions: Optional[Callable] = None,
) -> str:
    """Greedy generation with NO KV cache: each new token is one ``mt.trace`` of the whole sequence so far
    (plain argmax).

    ``interventions`` (an ``src.nrgs.functional.Intervention``, or any callable ``iv(mt)``) runs inside every
    one of those traces, so an edit "at the last token" hits each step's new position while every earlier
    position is recomputed without it (the obstacle OV boost relies on this; a KV-cached decode would keep the
    edited keys / values of earlier steps instead).

    Args:
        mt: ModelandTokenizer.
        input_ids: (1, seq) prompt token ids.
        attention_mask: (1, seq) attention mask.
        max_new_tokens: Maximum tokens to generate.
        stop_ids: Token ids that halt generation; the stop token is kept (and dropped by the decode when it is
            special). ``None`` (default) = ``tokenizer.eos_token_id``; an empty collection never stops early.
        interventions: edits applied in every forward, ``interventions(mt)`` inside the trace.

    Returns:
        Decoded generated string (prompt stripped, special tokens skipped, right-stripped).
    """
    _stop = [mt.tokenizer.eos_token_id] if stop_ids is None else stop_ids
    generated = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            with mt.trace(dict(input_ids=input_ids, attention_mask=attention_mask)):
                if interventions:
                    interventions(mt)
                logits = mt.logits[0, -1].save()
            next_tok = int(logits.argmax())
            generated.append(next_tok)
            if next_tok in _stop:
                break
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_tok]], device=input_ids.device)], dim=1
            )
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (1, 1), dtype=attention_mask.dtype, device=attention_mask.device
                    ),
                ],
                dim=1,
            )
    return mt.tokenizer.decode(generated, skip_special_tokens=True).rstrip()


def headpatched_generate(
    mt,
    dest_inputs: dict,
    src_o_proj_inps: torch.Tensor,
    mask,
    mask_layers: Optional[list[int]] = None,
    max_new_tokens: int = 64,
    n_steer_tokens: Optional[int] = None,
    alpha: float = 1.0,
) -> list[str]:
    """Greedy generation with head-replacement patching, batched over sources.

    ``src_o_proj_inps`` is ``(B, n_layers, num_heads * head_dim)``: B generations sharing ONE
    destination prompt (``dest_inputs`` is a single tokenized prompt, tiled internally). Returns
    ``list[str]`` of length B.

    Runs on ``mt.generate`` + ``tracer.iter``. At each patched step, for every mask row ``li``
    (block ``mask_layers[li]``) and every head ``h`` with ``mask[li, h] != 0``, the last
    position of ``mt.layers[l].self_attn.attention_premix`` (the heads at ``o_proj``'s input) is
    set to ``alpha * mask[li, h] * src_h + (1 - mask[li, h]) * cur_h`` in float32, ``cur_h`` being
    the live value (after the earlier blocks' edits). Step 0 is the prompt call (its last token);
    each decode step patches its one new token, whose K/V then carry the edit through the cache.

    ``n_steer_tokens`` counts *forward passes*, with the prompt call as the first;
    ``n_steer_tokens=1`` patches only the prompt call. ``None`` patches every step. (This is one
    more than ``generate_with_patch``'s count, which leaves the prompt call out.)

    Decoding is plain greedy argmax with a stop at ``tokenizer.eos_token_id``: the checkpoint's
    ``repetition_penalty`` and its other EOS ids are overridden, as in the hand-written decode loop
    this replaces. (``head_additive_generate`` keeps the checkpoint's ``generation_config``.)

    Args:
        mt: ModelandTokenizer.
        dest_inputs: Tokenized destination prompt dict (input_ids, attention_mask), batch 1.
        src_o_proj_inps: ``(B, n_layers, num_heads * head_dim)`` source o_proj inputs.
        mask: HeadMask with per-head interpolation weights.
        mask_layers: the ordered key decoding the mask's ROWS (row ``li`` <-> ``mask_layers[li]``),
            also matching ``src_o_proj_inps``' layer axis. ``None`` (default) = identity.
        max_new_tokens: Maximum tokens to generate per item.
        n_steer_tokens: Apply a fresh patch only on the first ``n_steer_tokens`` forward passes.
        alpha: source-side interpolation weight.
    """
    src = src_o_proj_inps
    if src.dim() != 3:
        raise ValueError(
            f"src_o_proj_inps must be (B, n_layers, n_embd); got {tuple(src.shape)}"
        )
    # `mask_layers` is the ordered key that decodes the mask's ROWS (row li <-> mask_layers[li]).
    # It is NOT "which layers to intervene at": slicing it silently shifts every row.
    if mask_layers is None:
        mask_layers = list(range(mask.values.shape[0]))
    if len(mask_layers) != src.shape[1] or len(mask_layers) != mask.values.shape[0]:
        raise ValueError(
            f"mask_layers/source/mask disagree: len(mask_layers)={len(mask_layers)}, "
            f"src layer axis={src.shape[1]}, mask rows={mask.values.shape[0]}. They align POSITIONALLY "
            f"(row li <-> mask_layers[li]), so a mismatch SHIFTS every row rather than failing."
        )
    B = src.shape[0]
    n_heads, head_dim = mt.num_heads, mt.head_dim
    inputs = dict(
        input_ids=dest_inputs["input_ids"].expand(B, -1).clone(),
        attention_mask=dest_inputs["attention_mask"].expand(B, -1).clone(),
    )
    L_prompt = inputs["input_ids"].shape[1]

    # Per block, in forward order: the heads its mask row moves (m != 0; the formula leaves a head
    # with m == 0 as it is), their weights and their source slices, on the block's device.
    mask_vals = mask.values.detach()
    plan = []
    for li, layer in sorted(enumerate(mask_layers), key=lambda t: t[1]):
        heads = torch.nonzero(mask_vals[li]).flatten()
        if heads.numel() == 0:
            continue
        attn = mt.layers[layer].self_attn
        dev = attn.o_proj.weight.device
        m = mask_vals[li, heads].float().to(dev).view(1, -1, 1)  # (1, k, 1)
        s = src[:, li].float().to(dev).view(B, n_heads, head_dim)[:, heads.to(dev)]  # (B, k, hd)
        plan.append((attn, heads.to(dev), m, s))
    n_steps = max_new_tokens if n_steer_tokens is None else min(max_new_tokens, n_steer_tokens)

    with torch.no_grad(), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=_LOOP_CUT_SHORT, category=UserWarning)
        with mt.generate(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.0,
            eos_token_id=mt.tokenizer.eos_token_id,
        ) as tracer:
            with tracer.invoke(inputs):
                if plan and n_steps > 0:
                    for step in tracer.iter[:n_steps]:
                        for attn, heads, m, s in plan:
                            premix = attn.attention_premix  # (B, seq, n_heads, head_dim)
                            cur = premix[:, -1, heads].float()  # (B, k, head_dim)
                            premix[:, -1, heads] = (alpha * m * s + (1 - m) * cur).to(premix.dtype)
            # Its own invoke, so an early EOS that cuts the step loop short cannot drop it.
            with tracer.invoke():
                out = tracer.result.save()

    return [
        mt.tokenizer.decode(row[L_prompt:], skip_special_tokens=True).rstrip()
        for row in out
    ]


def head_additive_generate(
    mt,
    dest_inputs: dict,
    steering_vecs: torch.Tensor,
    layers: list[int],
    max_new_tokens: int,
    n_steer_tokens: Optional[int] = None,
    alpha: float = 0.1,
) -> list[str]:
    """Greedy generation with a norm-relative ADDITIVE residual-stream injection — the counterpart
    of :func:`headpatched_generate`, which replaces per-head o_proj inputs instead.

    ``steering_vecs`` is ``(B, n_embd)``: B generations sharing ONE destination prompt, returns
    ``list[str]`` of length B. The same vector is injected into ``mt.layers[l].layer_output`` at
    every layer in ``layers``, at the last token of the prompt call and of every decode step:
    ``h + alpha * ||h|| * g/||g||``. ``alpha`` is "add this fraction of the local residual norm
    along the goal direction", which is scale-invariant across head sets, bands and models.
    Normalisation happens HERE, so callers pass the raw aggregated vector. Runs through
    :func:`src.functional.generate_with_patch` (greedy, the checkpoint's ``generation_config``
    otherwise).

    Args:
        mt: ModelandTokenizer.
        dest_inputs: Tokenized destination prompt dict, batch 1 (tiled internally).
        steering_vecs: ``(B, n_embd)`` aggregated head outputs. Raw, unnormalised.
        layers: layer indices to inject at (the steer band), e.g. ``range(35, 46)``.
        max_new_tokens: Maximum tokens to generate per item.
        n_steer_tokens: how many decode steps apply the patch, after the prompt call (which
            always does). ``None`` = every step.
        alpha: injection coefficient.
    """
    if steering_vecs.dim() != 2:
        raise ValueError(
            f"steering_vecs must be (B, n_embd); got {tuple(steering_vecs.shape)}"
        )
    # Hand the patch site a UNIT direction; the magnitude comes from ||h|| there.
    v = steering_vecs
    v = alpha * (v / v.float().norm(dim=-1, keepdim=True).clamp_min(1e-6).to(v.dtype))
    B = v.shape[0]

    tiled = TokenizerOutput(
        dict(
            input_ids=dest_inputs["input_ids"].expand(B, -1).clone(),
            attention_mask=dest_inputs["attention_mask"].expand(B, -1).clone(),
        )
    )
    patches = [
        PatchSpec(location=(li, -1), patch=v, strategy="add_norm", on_pass="all")
        for li in layers
    ]
    gens = generate_with_patch(
        mt,
        tiled,
        max_new_tokens=max_new_tokens,
        patches=patches,
        n_steer_tokens=n_steer_tokens,
        do_sample=False,
        remove_prefix=True,
        skip_special_tokens=True,
    )
    return list(gens)


# ---------------------------------------------------------------------------
# GoalScope destination prompts
# ---------------------------------------------------------------------------

# Content of the GoalScope destination prompt.
GOALSCOPE_SYS = "You are a helpful assistant proficient in programming."
GOALSCOPE_QUESTION = "What are you doing at this line of the code. Keep your answer as brief as possible."


def _build_goalscope_prompt(mt) -> str:
    """Build the GoalScope destination prompt ("What are you doing at this line?")."""
    messages = [
        {"role": "system", "content": GOALSCOPE_SYS},
        {"role": "user", "content": GOALSCOPE_QUESTION},
        {"role": "assistant", "content": ""},
    ]
    return chat_prompt(mt, messages, add_generation_prompt=False, continue_final_message=True)


# GoalScope destination prompt for the causality read-out: an information-less C++ function stub. The
# goal encoding is steered in on top of it; the model then continues the C++ body cross-lingually.
CAUSAL_PREFIX = "bool check(auto value){"
CAUSAL_INSTRUCTION = (
    "You are implementing a simple C++ function that checks for a condition. "
    "Just write the function body, without any comments or explanations. Assume all the necessary "
    "headers are already included."
)


def _build_causality_prompt(mt) -> str:
    """Build the GoalScope *causality* destination prompt — an information-less C++ function stub
    (assistant turn == ``CAUSAL_PREFIX``). Cross-language counterpart of :func:`_build_goalscope_prompt`."""
    messages = [
        {"role": "system", "content": GOALSCOPE_SYS},
        {"role": "user", "content": CAUSAL_INSTRUCTION},
        {"role": "assistant", "content": CAUSAL_PREFIX},
    ]
    return chat_prompt(mt, messages, add_generation_prompt=False, continue_final_message=True)


class OVReader:
    """OV-write residual probe: is the goal already in the residual, in the goal heads' OV-write direction?

    For a residual ``h`` (the stream after some layer ``L``, at a pivot token), a goal head ``(l, j)`` would
    write ``W_O_j W_V_kv (g ⊙ h)`` into the residual if ``h`` were its value source — the full OV transform,
    NOT a projection: the head reads its value on the RMSNorm'd residual (per-dim gain ``g`` folded; the
    direction-preserving ``1/rms`` scalar is dropped), maps it through ``W_V`` (GQA kv-head ``kv = j // group``),
    then writes it through ``W_O``. Aggregating the unit OV-writes over the readout heads gives a residual
    DIRECTION ``h_ov``; steering the GoalScope probe with it (``head_additive_generate`` / ``additive_norm``)
    verbalizes the goal iff ``h`` already carries it in that write direction.

    The per-head factors ``(W_O_j, W_V_kv, g)`` are computed once in ``__init__`` and cached in ``readers``.
    They are read off nnterp's envoys: ``mt.layers[l].self_attn.o_proj`` / ``.v_proj`` and the block's
    pre-attention norm ``mt.layers[l].input_layernorm``. A family whose attention reads the residual
    without a pre-attention norm (OLMo-3 is post-norm) has no ``g`` to fold, and is refused.
    """

    def __init__(self, mt, readout_heads: list[tuple[int, int]]):
        self.mt = mt
        self.readout_heads = list(readout_heads)
        self.n_kv = mt.num_kv_heads
        self.group = mt.num_heads // self.n_kv  # GQA group size
        self.readers = self._build_readers()  # per-head (W_O_j, W_V_kv, g), CPU float32
        self._probe_inputs = {}  # {mode: tokenized GoalScope probe}, built lazily

    def _build_readers(self) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Per goal head ``(l, j)``: ``(W_O_j, W_V_kv, g)`` — small factors of the OV map
        ``W_O_j W_V_kv diag(g)``, kept as CPU float32 and applied lazily in :meth:`resid_ov_vec`."""
        hd = self.mt.head_dim
        readers = []
        for (layer, j) in tqdm(self.readout_heads, desc="Building OV readers"):
            block = self.mt.layers[layer]
            norm = getattr(block, "input_layernorm", None)
            if norm is None:
                raise ValueError(
                    f"OVReader folds the pre-attention norm's gain into each head's OV map, but "
                    f"{self.mt.family_name!r} blocks have no pre-attention norm (input_layernorm): "
                    f"the attention reads the raw residual and the block norms after it (OLMo-3 is "
                    f"post-norm). The OV map without a gain is a different probe; not supported."
                )
            kv = j // self.group
            Wo = block.self_attn.o_proj.weight[:, j * hd : (j + 1) * hd].detach().float().cpu()
            Wv = block.self_attn.v_proj.weight[kv * hd : (kv + 1) * hd, :].detach().float().cpu()
            g = norm.weight.detach().float().cpu()
            readers.append((Wo, Wv, g))
        return readers

    def resid_ov_vec(self, h: torch.Tensor) -> torch.Tensor:
        """``h_ov = mean_j unit(W_O_j W_V_kv (g ⊙ h))`` — the aggregated OV-write direction for value ``h``.
        ``h`` is any ``(n_embd,)`` residual (moved to CPU float32); returns ``(n_embd,)`` CPU float32."""
        h = h.detach().float().cpu()
        vecs = []
        for Wo, Wv, g in self.readers:
            h_ov = Wo @ (Wv @ (g * h))
            vecs.append(h_ov / h_ov.norm().clamp_min(1e-8))
        return torch.stack(vecs).mean(0)

    def probe_inputs(self, mode: Literal["faithfulness", "causality"] = "faithfulness"):
        """Tokenized GoalScope destination prompt for ``mode`` (cached per mode). ``"faithfulness"`` =
        the NL "what are you doing at this line?" probe; ``"causality"`` = the C++ function stub."""
        if mode not in self._probe_inputs:
            builder = _build_goalscope_prompt if mode == "faithfulness" else _build_causality_prompt
            self._probe_inputs[mode] = prepare_input(prompts=builder(self.mt), tokenizer=self.mt)
        return self._probe_inputs[mode]

    def apply_probe(
        self,
        latents: dict,
        *,
        mode: Literal["faithfulness", "causality"] = "faithfulness",
        steer_layers: Optional[list[int]] = None,
        max_new_tokens: int = 64,
        alpha: float = 0.15,
        batch_size: int = 16,
    ) -> dict:
        """Probe already-extracted residual latents, in chunks of ``batch_size``.

        ``latents`` is a ``{key: (n_embd,) residual}`` dict; returns ``{key: readout}``. Each residual
        is turned into its OV-write steering vector (:meth:`resid_ov_vec`) and steered (additive_norm)
        into the GoalScope probe for ``mode`` at ``steer_layers`` (default ``range(35, 46)``), one
        ``head_additive_generate`` call per chunk so memory stays bounded.
        """
        steer_layers = list(range(35, 46)) if steer_layers is None else list(steer_layers)
        keys = list(latents)
        vecs = [latents[k] for k in keys]
        pin = self.probe_inputs(mode)
        outs: list[str] = []
        for s in range(0, len(vecs), batch_size):
            steer = torch.stack([self.resid_ov_vec(h) for h in vecs[s : s + batch_size]])  # (b, n_embd) CPU f32
            outs.extend(head_additive_generate(
                self.mt,
                pin,
                steer,
                layers=steer_layers,
                max_new_tokens=max_new_tokens,
                alpha=alpha,
            ))
        return dict(zip(keys, outs))

    def apply_probe_on_prompt(
        self,
        prompt: str | TokenizerOutput,
        pivot: int,
        probe_layers: list[int],
        **kwargs,
    ) -> dict[int, str]:
        """Read the residual after each ``probe_layer`` (``mt.layers[L].layer_output``) at ``pivot`` in
        ONE trace, then :meth:`apply_probe` on those latents. Returns ``{L: readout}``. ``kwargs`` (mode,
        steer_layers, max_new_tokens, alpha, batch_size) are forwarded to :meth:`apply_probe`."""
        inp = prepare_input(prompts=prompt, tokenizer=self.mt) if isinstance(prompt, str) else prompt
        if "offset_mapping" in inp:
            inp.pop("offset_mapping")
        probe_layers = list(probe_layers)
        blocks = {L: self.mt.layers[L] for L in sorted(set(probe_layers))}
        saved = {}
        with torch.no_grad(), self.mt.trace(dict(inp)):
            for L, block in blocks.items():  # ascending: forward order
                saved[L] = block.layer_output[0, pivot].save()
        latents = {L: saved[L] for L in probe_layers}
        return self.apply_probe(latents, **kwargs)


def apply_goalscope(
    prompt: str | TokenizerOutput,
    *,
    mt,
    selected_heads: list[tuple[int, int]],
    strategy: Literal["additive_norm", "replace"] = "additive_norm",
    steer_layer_range: tuple[int, int] = (35, 46),
    batch_size: int = 16,
    alpha: float = 0.1,
    max_new_tokens: int = 64,
    n_steer_tokens: Union[int, Literal["all"]] = "all",
    goalscope_prompt: Optional[str] = None,
    include_noncode: bool = False,
    token_positions: Optional[list[int]] = None,
    interventions: Optional[Callable] = None,
) -> list[dict]:
    """Run GoalScope on a single prompt.

    Extracts goal signals from the ``<code>`` segment of ``prompt`` and injects
    them into the GoalScope probe ("What are you doing at this line?") to produce
    a verbalization per token position.

    Two intervention strategies:

    - ``"additive_norm"`` (default): aggregate the selected heads' o_proj
      contributions into a single residual-stream vector ``g_enc`` per source
      position and add it norm-relatively, ``h + alpha * ||h|| * g_enc/||g_enc||``,
      to the residual stream at every layer in ``steer_layer_range`` (last
      token).  ``alpha`` is "add this fraction of the local residual norm along
      the goal direction", so it is comparable across head sets, layer bands and
      models.  Uses ``generate_with_patch``.
    - ``"replace"``: surgically replace the selected heads' o_proj inputs at the
      destination's last token with the corresponding source o_proj inputs.
      Layers come from ``selected_heads``; ``steer_layer_range`` is ignored.
      ``alpha`` controls the source-side weight inside the per-head
      interpolation ``alpha * mask * src + (1 - mask) * cascade``.  Uses
      ``headpatched_generate``.

    Args:
        prompt: Full text (prompt + generation) to analyse, OR a pre-tokenized
            ``TokenizerOutput``.  A string is tokenized here and the ``<code>``
            segment is auto-located.  A ``TokenizerOutput`` is used as-is — token
            positions then align exactly with the caller's own tokenization and
            the prompt is not re-tokenized, but the whole sequence is scoped (no
            string to find ``<code>`` in), so pass ``token_positions`` to restrict.
        mt: ModelandTokenizer.
        selected_heads: ``(layer, head_idx)`` pairs from the trained head mask.
        strategy: ``"additive_norm"`` (default) or ``"replace"``.  See above.
        steer_layer_range: ``(start, end)`` — layers ``range(start, end)``
            receive the additive injection.  **additive_norm only.**
        batch_size: Token positions per generation call.
        alpha: Steering strength.  additive_norm: added fraction of ``||h||``.
            Replace: source-side interpolation weight.
        max_new_tokens: Max tokens generated per position.
        n_steer_tokens: How many generate steps apply a fresh patch.
            ``"all"`` (default) → patch every step.  An integer ``k`` counts
            differently per strategy (each keeps its original convention):
            ``"replace"`` patches the first ``k`` forward passes, the prompt call
            counting as 1; ``"additive_norm"`` patches the prompt call and then
            ``k`` decode steps, ``k + 1`` forward passes in all.
        goalscope_prompt: Override the default GoalScope dest prompt.
        include_noncode: When ``True``, also include the non-code tokens (the
            prompt text outside the ``<code>…</code>`` segment — e.g. the prefix
            / instruction) in the returned results.  These positions are **not**
            goal-scoped: no steering source is extracted and no generation is run
            for them.  Extraction / generation still run only over the code segment.
        token_positions: Absolute token indices (in the tokenization of ``prompt``) to
            extract + generate at.  Must fall inside the ``<code>`` segment.  ``None``
            (default) uses every code-segment token.  Restricting to a handful of
            positions (e.g. the anchors) skips generation for all others and is much
            faster.
        interventions: optional ``interventions(mt)``, called inside the SOURCE trace that reads
            ``selected_heads`` (the goal heads), before the reads; see
            :func:`src.goals.head_loc_attn.get_o_proj_inputs_batch` for the composition rule. An
            intervention on a DIFFERENT head set (e.g. transplanted at the pivot) then reaches the
            goal-head signal that is verbalized. ``None`` (default) → no intervention. Its positions
            are absolute indices into ``prompt``'s tokenization, exactly like ``token_positions``.
            The readout generation never runs under it. The NRGS interventions
            (:class:`src.nrgs.functional.Intervention`) are such callables.

    Returns:
        List of dicts with keys ``"pos"``, ``"token"``, ``"patched_gen"`` and
        ``"scoped"`` (``True`` for code-segment tokens; ``False`` for non-code
        tokens when ``include_noncode=True``).  Code-segment entries come first
        in generation order; with ``include_noncode`` the full list is sorted by
        ``pos`` so the prompt reads left-to-right.
    """
    if strategy not in ("additive_norm", "replace"):
        raise ValueError(
            f"strategy must be 'additive_norm' or 'replace', got {strategy!r}"
        )

    # Translate n_steer_tokens="all" → None for the underlying APIs
    n_steer = None if n_steer_tokens == "all" else int(n_steer_tokens)

    if goalscope_prompt is None:
        goalscope_prompt = _build_goalscope_prompt(mt)

    # Source tokenization — accept a raw ``str`` (tokenized here) or a pre-tokenized
    # ``TokenizerOutput`` (used as-is, so token positions align EXACTLY with the caller's own
    # tokenization and the prompt is not re-tokenized).  With a ``TokenizerOutput`` there is no
    # string to locate ``<code>`` in, so the whole sequence is scoped — pass ``token_positions`` to
    # restrict it.
    if isinstance(prompt, TokenizerOutput):
        src_tokenized = prompt
        if "offset_mapping" in src_tokenized:
            src_tokenized.pop("offset_mapping")
        src_ids = src_tokenized["input_ids"][0].tolist()
        text_tokens = [mt.tokenizer.decode([t]) for t in src_ids]
        code_start, code_end = 0, len(src_ids)
    else:
        src_tokenized = prepare_input(
            prompts=prompt, tokenizer=mt, return_offsets_mapping=True
        )
        src_offsets = src_tokenized.pop("offset_mapping")[0]
        src_ids = src_tokenized["input_ids"][0].tolist()
        text_tokens = [mt.tokenizer.decode([t]) for t in src_ids]

        # Find <code> … </code> segment; fall back to full sequence
        try:
            code_start, _ = find_token_range(
                string=prompt,
                substring="<code>",
                tokenizer=mt.tokenizer,
                occurrence=1,
                offset_mapping=src_offsets,
            )
            if prompt.count("</code>") > 1:
                _, code_end = find_token_range(
                    string=prompt,
                    substring="</code>",
                    tokenizer=mt.tokenizer,
                    occurrence=1,
                    offset_mapping=src_offsets,
                )
            else:
                code_end = len(src_ids)
        except Exception:
            code_start, code_end = 0, len(src_ids)

    # print(f"Scoping tokens from {code_start} to {code_end}")

    seg_positions = list(range(code_start, code_end))
    if token_positions is not None:
        # Restrict extraction + generation to the requested positions (must lie inside the scoped
        # segment). Steering-source extraction and generation both key off seg_positions, so this is
        # all that's needed to skip the other tokens — much faster.
        allowed = set(seg_positions)
        seg_positions = [int(p) for p in token_positions if int(p) in allowed]
        if not seg_positions:
            raise ValueError(
                "token_positions is empty after intersecting with the scoped segment "
                f"[{code_start}, {code_end})"
            )
    n_seg = len(seg_positions)

    # Tokenize GoalScope dest once
    dest_tok = prepare_input(prompts=goalscope_prompt, tokenizer=mt)
    dest_ids = dest_tok["input_ids"]
    dest_attn = dest_tok["attention_mask"]

    results: list[dict] = []

    # -----------------------------------------------------------------
    # Strategy-specific source extraction
    # -----------------------------------------------------------------
    if strategy == "additive_norm":
        # Aggregate selected heads into one residual-stream vector per position.
        # collect_head_steering_vecs returns (batch_size, n_seg, n_embd);
        # src_tokenized here has batch_size=1, so squeeze the leading dim.
        steering_vecs = collect_head_steering_vecs(
            mt=mt,
            src_tokenized=src_tokenized,
            selected_heads=selected_heads,
            token_positions=seg_positions,
            interventions=interventions,
        )[0]  # (n_seg, n_embd)

        # additive_norm's unit-direction normalisation happens inside
        # head_additive_generate, so `steering_vecs` stays raw here.

        layer_start, layer_end = steer_layer_range
        steer_layers = list(range(layer_start, layer_end))

    else:  # strategy == "replace"
        # Layers come from selected_heads; build a HeadMask over them.
        layers_replace = sorted({layer for (layer, _) in selected_heads})
        n_heads_model = mt.num_heads
        mask_module = HeadMask(
            n_layers_tested=len(layers_replace), n_heads=n_heads_model
        )
        with torch.no_grad():
            mask_module.values.zero_()
            for layer, h in selected_heads:
                li = layers_replace.index(layer)
                mask_module.values[li, h] = 1.0

        # Extract per-layer o_proj inputs at every segment position.
        # Shape per layer: (1, n_seg, n_embd); stack → (n_seg, n_layers, n_embd).
        src_o_inputs = get_o_proj_inputs_batch(
            mt,
            src_tokenized,
            layers_replace,
            token_positions=seg_positions,
            interventions=interventions,
        )
        src_head_vals_all = torch.stack(
            [src_o_inputs[layer][0] for layer in layers_replace], dim=1
        )  # (n_seg, n_layers, n_embd)

    # -----------------------------------------------------------------
    # Batched generation
    # -----------------------------------------------------------------
    for chunk_start in tqdm(range(0, n_seg, batch_size), desc=f"GoalScope[{strategy}]"):
        chunk_end = min(chunk_start + batch_size, n_seg)
        B = chunk_end - chunk_start

        tiled = TokenizerOutput(
            data=dict(
                input_ids=dest_ids.expand(B, -1).clone(),
                attention_mask=dest_attn.expand(B, -1).clone(),
            )
        )

        if strategy == "additive_norm":
            gens = head_additive_generate(
                mt=mt,
                dest_inputs=tiled,
                steering_vecs=steering_vecs[chunk_start:chunk_end],  # (B, n_embd), raw
                layers=steer_layers,
                max_new_tokens=max_new_tokens,
                n_steer_tokens=n_steer,
                alpha=alpha,
            )
        else:  # replace
            src_chunk = src_head_vals_all[
                chunk_start:chunk_end
            ]  # (B, n_layers, n_embd)
            gens = headpatched_generate(
                mt=mt,
                dest_inputs=tiled,
                src_o_proj_inps=src_chunk,
                mask=mask_module,
                mask_layers=layers_replace,
                max_new_tokens=max_new_tokens,
                n_steer_tokens=n_steer,
                alpha=alpha,
            )

        for i in range(B):
            results.append(
                {
                    "pos": seg_positions[chunk_start + i],
                    "token": text_tokens[seg_positions[chunk_start + i]],
                    "patched_gen": gens[i],
                    "scoped": True,
                }
            )

    # Optionally fold in the non-code tokens (everything outside [code_start, code_end))
    # as display-only, non-scoped context — no source extraction / generation for them.
    if include_noncode:
        seg_set = set(seg_positions)
        noncode = [
            {"pos": p, "token": text_tokens[p], "patched_gen": "", "scoped": False}
            for p in range(len(src_ids))
            if p not in seg_set
        ]
        results = sorted(results + noncode, key=lambda r: r["pos"])

    return results
