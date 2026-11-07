"""Interactive qualitative GoalScope demo (paper-style callouts).

Turns the output of :func:`src.goals.goalscope.apply_goalscope` — a ``list[dict]`` with keys
``pos`` / ``token`` / ``patched_gen`` / ``scoped`` — into a self-contained HTML page in the style of
``Overleaf-draft/REF/qualitative_demo.png``: the source code on the right (syntax-tinted, monospace),
and, on the left, italic-quote callout boxes connected by curved leader lines to pink anchor dots on
the chosen tokens.

Interaction: **click a scoped token** to pin its read-out — a numbered callout appears on the left,
the token is boxed and gets an anchor dot, and a leader line is drawn. Click again to unpin. Multiple
tokens can be pinned at once. Hovering a scoped token shows a transient preview.

Usage::

    from src.goals.goalscope import apply_goalscope
    from src.goals.goalscope_viz import goalscope_qualitative_html, show_goalscope_qualitative
    res = apply_goalscope(full_text, mt=mt, selected_heads=heads, strategy="additive_norm",
                          alpha=0.1, include_noncode=True, show=False)
    show_goalscope_qualitative(res, title="stable_marriage", height=760)      # inline (iframe)
    goalscope_qualitative_html(res, save_path="figures/qual_demo.html")        # standalone .html
"""

from __future__ import annotations

import html as _html
import json
import uuid
from typing import Optional

# Lightweight Python token tinting (cosmetic — tokens are subwords, matched on stripped text).
_KEYWORDS = {
    "def", "return", "while", "for", "if", "elif", "else", "in", "not", "and", "or", "break",
    "continue", "pass", "import", "from", "class", "with", "as", "lambda", "yield", "try",
    "except", "finally", "raise", "is", "assert", "global", "nonlocal", "del", "await", "async",
}
_CONSTS = {"True", "False", "None"}
_BUILTINS = {
    "range", "len", "print", "int", "str", "float", "bool", "list", "dict", "set", "tuple",
    "enumerate", "zip", "map", "filter", "sorted", "sum", "min", "max", "abs", "any", "all",
    "sorted", "reversed", "round", "type", "isinstance",
}


def _tok_class(text: str) -> str:
    s = text.strip()
    if s in _KEYWORDS:
        return "kw"
    if s in _CONSTS:
        return "cn"
    if s in _BUILTINS:
        return "bi"
    if s and (s.isdigit() or (s.replace(".", "", 1).isdigit())):
        return "num"
    return ""


_CSS = """
:root{
  --bg:#ffffff; --code:#1f2328; --dim:#c4c8cf; --muted:#6b7280;
  --kw:#a626a4; --cn:#0b76c4; --bi:#3aa39f; --num:#0b76c4; --fn:#6f42c1;
  --anchor:#f2708a; --anchor-line:#9aa1ab; --box:#d0d5dd; --hi:#fff2b8; --pin:#f2708a;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--code);
  font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
.gs-wrap{position:relative;display:flex;align-items:stretch;gap:0;padding:22px 26px 40px;--leftw:46%;}
.gs-left{position:relative;flex:0 0 var(--leftw);min-height:100%;}
.gs-divider{flex:0 0 14px;align-self:stretch;position:relative;cursor:col-resize;z-index:4;}
.gs-divider::before{content:"";position:absolute;left:50%;top:0;bottom:0;width:1px;background:#e5e7eb;transform:translateX(-50%);transition:background .1s,width .1s;}
.gs-divider:hover::before,.gs-divider.drag::before{background:#9aa1ab;width:2px;}
.gs-right{flex:1 1 auto;padding-left:20px;min-width:0;}
.gs-title{font-size:13px;color:var(--muted);letter-spacing:.02em;margin:0 0 10px;text-transform:uppercase;}
pre.gs-code{margin:0;font-family:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,"Liberation Mono",monospace;
  font-size:15px;line-height:1.7;white-space:pre-wrap;word-break:break-word;}
.gs-tok{border-radius:4px;padding:0 .5px;}
.gs-tok.scoped{cursor:pointer;transition:background .12s;}
.gs-tok.scoped:hover{background:var(--hi);}
.gs-tok.pinned{background:#ffe08a;box-shadow:0 0 0 1.5px var(--pin);border-radius:4px;}
.gs-tok.noncode{color:var(--dim);}
.gs-tok.kw{color:var(--kw);font-weight:600}
.gs-tok.cn{color:var(--cn);font-weight:600}
.gs-tok.bi{color:var(--bi)}
.gs-tok.num{color:var(--num)}
svg.gs-links{position:absolute;inset:0;width:100%;height:100%;pointer-events:none;z-index:2;overflow:visible;}
.gs-callout{position:absolute;left:20px;max-width:calc(var(--leftw) - 48px);background:#fff;border:1px solid var(--box);
  border-radius:12px;padding:11px 14px;box-shadow:0 1px 3px rgba(16,24,40,.07);z-index:3;
  font-size:15px;line-height:1.45;font-style:italic;color:#111827;}
.gs-callout .idx{position:absolute;left:-11px;top:-11px;width:22px;height:22px;border-radius:50%;
  background:var(--pin);color:#fff;font-style:normal;font-weight:700;font-size:12px;
  display:flex;align-items:center;justify-content:center;box-shadow:0 1px 2px rgba(0,0,0,.2);}
.gs-hint{position:absolute;top:22px;right:26px;font-size:12px;color:var(--muted);font-style:italic;}
.gs-tip{position:fixed;pointer-events:none;z-index:9;max-width:320px;background:#111827;color:#fff;
  font-size:13px;line-height:1.4;font-style:italic;padding:8px 11px;border-radius:8px;opacity:0;
  transition:opacity .1s;box-shadow:0 4px 14px rgba(0,0,0,.25);}
.gs-empty{color:var(--muted);font-size:14px;font-style:italic;padding-top:6px;}
""".strip()


# The interactive layer. {CID} is substituted with a per-render container id; all braces here are JS.
_JS = r"""
(function(){
  const CID = "__CID__";
  const root = document.getElementById(CID);
  if(!root) return;
  const TOK = __TOKENS_JSON__;                 // [{pos,text,readout,scoped}]
  const readouts = {}; TOK.forEach(t=>{ if(t.scoped) readouts[t.pos]=t.readout; });
  const left  = root.querySelector(".gs-left");
  const right = root.querySelector(".gs-right");
  const svg   = root.querySelector("svg.gs-links");
  const tip   = root.querySelector(".gs-tip");
  const pinned = [];                           // ordered list of pos (pin order)

  function esc(s){ return (s||"").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }

  // hover preview
  root.querySelectorAll(".gs-tok.scoped").forEach(el=>{
    el.addEventListener("mousemove", e=>{
      const p = +el.dataset.pos;
      tip.textContent = "“"+ (readouts[p]||"") +"”";
      tip.style.opacity = .98;
      tip.style.left = Math.min(e.clientX+14, window.innerWidth-336) + "px";
      tip.style.top  = (e.clientY+16) + "px";
    });
    el.addEventListener("mouseleave", ()=>{ tip.style.opacity = 0; });
    el.addEventListener("click", ()=>{
      const p = +el.dataset.pos;
      const i = pinned.indexOf(p);
      if(i>=0){ pinned.splice(i,1); el.classList.remove("pinned"); }
      else    { pinned.push(p);     el.classList.add("pinned"); }
      render();
    });
  });

  function render(){
    // clear callouts + links
    root.querySelectorAll(".gs-callout").forEach(n=>n.remove());
    while(svg.firstChild) svg.removeChild(svg.firstChild);
    const empty = left.querySelector(".gs-empty"); if(empty) empty.remove();
    if(pinned.length===0){
      const e=document.createElement("div"); e.className="gs-empty";
      e.textContent="Click a highlighted token on the right to pin its read-out here.";
      left.appendChild(e); return;
    }
    // Single origin for callouts (children of root, absolute) + svg user space + measurements:
    // the svg is position:absolute inset:0, so its box IS root's padding box == the abs-positioning
    // containing block. Measure everything relative to it.
    const O = svg.getBoundingClientRect();
    // sort pins by the token's vertical position so callouts read top-to-bottom
    const order = pinned.map(p=>{
      const t = right.querySelector('.gs-tok[data-pos="'+p+'"]');
      const r = t.getBoundingClientRect();
      return {p, t, ty:(r.top+r.bottom)/2 - O.top, tx:r.left - O.left};
    }).sort((a,b)=>a.ty-b.ty);

    // create callouts, then position with simple vertical de-collision
    let lastBottom = 0;
    order.forEach((o,k)=>{
      const box=document.createElement("div"); box.className="gs-callout";
      const n = pinned.indexOf(o.p)+1;                 // stable pin number
      box.innerHTML='<span class="idx">'+n+'</span>'
                    +'“'+esc(readouts[o.p])+'”';
      root.appendChild(box);
      const bh = box.offsetHeight;
      let top = Math.max(o.ty - bh/2, lastBottom + 12);
      box.style.top = top + "px";
      lastBottom = top + bh;
      o.box = box; o.n = n;
    });
    // draw leader lines + anchor dots (after layout is known)
    order.forEach(o=>{
      const br = o.box.getBoundingClientRect();
      const x1 = br.right - O.left, y1 = br.top - O.top + o.box.offsetHeight/2;
      const tr = o.t.getBoundingClientRect();
      const x2 = tr.left - O.left - 2, y2 = (tr.top+tr.bottom)/2 - O.top;
      const mx = (x1+x2)/2;
      const path=document.createElementNS("http://www.w3.org/2000/svg","path");
      path.setAttribute("d",`M ${x1} ${y1} C ${mx} ${y1}, ${mx} ${y2}, ${x2} ${y2}`);
      path.setAttribute("fill","none"); path.setAttribute("stroke","#9aa1ab");  // var() not valid in SVG attrs
      path.setAttribute("stroke-width","1.4");
      svg.appendChild(path);
      const c=document.createElementNS("http://www.w3.org/2000/svg","circle");
      c.setAttribute("cx",x2); c.setAttribute("cy",y2); c.setAttribute("r","4");
      c.setAttribute("fill","#f2708a"); c.setAttribute("stroke","#fff"); c.setAttribute("stroke-width","1");
      svg.appendChild(c);
    });
  }
  window.addEventListener("resize", render);

  // draggable splitter — resize the left/right panels; callouts + leader lines re-flow live.
  const divider = root.querySelector(".gs-divider");
  let dragging = false;
  divider.addEventListener("mousedown", e=>{
    dragging = true; divider.classList.add("drag");
    document.body.style.userSelect = "none"; e.preventDefault();
  });
  document.addEventListener("mousemove", e=>{
    if(!dragging) return;
    const wr = root.getBoundingClientRect();
    const cs = getComputedStyle(root);
    const padL = parseFloat(cs.paddingLeft)||0, padR = parseFloat(cs.paddingRight)||0;
    const content = wr.width - padL - padR;
    const w = e.clientX - (wr.left + padL);
    const frac = Math.min(0.80, Math.max(0.18, w/content));   // clamp 18%..80%
    root.style.setProperty("--leftw", (frac*100).toFixed(2)+"%");
    render();
  });
  document.addEventListener("mouseup", ()=>{
    if(dragging){ dragging=false; divider.classList.remove("drag");
      document.body.style.userSelect=""; render(); }
  });

  render();
})();
""".strip()


def goalscope_qualitative_html(
    results: list[dict],
    *,
    title: str = "GoalScope",
    highlight: bool = True,
    save_path: Optional[str] = None,
    container_id: Optional[str] = None,
) -> str:
    """Build a self-contained standalone HTML document from ``apply_goalscope`` output.

    Args:
        results: list of dicts with ``pos`` / ``token`` / ``patched_gen`` / ``scoped``. Pass the
            output of ``apply_goalscope(..., include_noncode=True, show=False)`` so the full source
            prompt is shown (non-code tokens are dimmed and not clickable).
        title: shown above the code panel.
        highlight: apply lightweight Python keyword/constant/builtin tinting.
        save_path: if given, write the standalone HTML doc there.
        container_id: DOM id (auto-generated if None).

    Returns:
        The standalone HTML document string.
    """
    cid = container_id or ("gs_" + uuid.uuid4().hex[:8])
    rows = sorted(results, key=lambda r: r["pos"])

    spans = []
    tokens_json = []
    for r in rows:
        pos = int(r["pos"])
        text = r.get("token", "")
        scoped = bool(r.get("scoped", True)) and bool((r.get("patched_gen") or "").strip())
        cls = ["gs-tok"]
        if scoped:
            cls.append("scoped")
        else:
            cls.append("noncode")
        if highlight:
            tc = _tok_class(text)
            if tc:
                cls.append(tc)
        # Newline tokens are invisible in <pre>; show a visible ↵ glyph before the real \n so the
        # span is still hoverable/clickable (matches the original goalscope visualization).
        tok_html = _html.escape(text).replace("\n", "↵\n")
        spans.append(
            f'<span class="{" ".join(cls)}" data-pos="{pos}">{tok_html}</span>'
        )
        tokens_json.append(
            {"pos": pos, "text": text, "scoped": scoped,
             "readout": (r.get("patched_gen") or "").strip() if scoped else ""}
        )

    code_html = "".join(spans)
    js = _JS.replace("__CID__", cid).replace(
        "__TOKENS_JSON__", json.dumps(tokens_json, ensure_ascii=False)
    )
    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_html.escape(title)}</title><style>{_CSS}</style></head>
<body>
<div class="gs-wrap" id="{cid}">
  <div class="gs-left"></div>
  <div class="gs-divider" title="drag to resize"></div>
  <div class="gs-right">
    <div class="gs-title">{_html.escape(title)}</div>
    <pre class="gs-code">{code_html}</pre>
  </div>
  <svg class="gs-links"></svg>
  <div class="gs-tip"></div>
</div>
<script>{js}</script>
</body></html>"""

    if save_path:
        with open(save_path, "w", encoding="utf-8") as fh:
            fh.write(doc)
    return doc


# ═══════════════════════════════════════════════════════════════════════════════
# Flow variant — for natural text (not code): callouts snap to a LEFT/RIGHT margin (two fixed
# columns flanking a narrower text column) or a BOTTOM margin (a reserved full-width strip below
# the whole text, shelf-packed under each token's own x-position) instead of floating on top of
# the text itself. All three are reserved space, never overlapping the paragraph -- same idea as
# the original fixed-left renderer, just on three sides instead of one. Each pinned callout gets a
# left/right/below toggle. Not full drag-anywhere positioning — see the module docstring note
# above goalscope_qualitative_flow_html for the tradeoff.
# ═══════════════════════════════════════════════════════════════════════════════

_CSS_FLOW = """
:root{
  --bg:#ffffff; --code:#1f2328; --dim:#c4c8cf; --muted:#6b7280;
  --anchor:#f2708a; --anchor-line:#9aa1ab; --box:#d0d5dd; --hi:#fff2b8; --pin:#f2708a;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--code);
  font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
.gsf-wrap{position:relative;max-width:1180px;margin:0 auto;padding:26px 0 40px;--marginLw:260px;--marginRw:260px;}
.gsf-row{display:flex;align-items:stretch;gap:0;}
.gsf-marginL{position:relative;flex:0 0 var(--marginLw);min-height:100%;}
.gsf-marginR{position:relative;flex:0 0 var(--marginRw);min-height:100%;}
.gsf-textcol{flex:1 1 auto;min-width:0;padding:0 26px;}
.gsf-div{flex:0 0 14px;align-self:stretch;position:relative;cursor:col-resize;z-index:4;}
.gsf-div::before{content:"";position:absolute;left:50%;top:0;bottom:0;width:1px;background:#e5e7eb;transform:translateX(-50%);transition:background .1s,width .1s;}
.gsf-div:hover::before,.gsf-div.drag::before{background:#9aa1ab;width:2px;}
.gsf-marginB{position:relative;margin-top:22px;padding-top:18px;border-top:1px solid #eceef1;min-height:1px;}
.gsf-title{font-size:13px;color:var(--muted);letter-spacing:.02em;margin:0 0 14px;text-transform:uppercase;}
.gsf-text{position:relative;font-family:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,"Liberation Mono",monospace;
  font-size:16px;line-height:2.1;white-space:pre-wrap;word-break:break-word;}
.gs-tok{border-radius:4px;padding:0 .5px;}
.gs-tok.scoped{cursor:pointer;transition:background .12s;background:#e2edfa;}
.gs-tok.scoped:hover{background:var(--hi);}
.gs-tok.pinned{background:#ffe08a;box-shadow:0 0 0 1.5px var(--pin);border-radius:4px;}
.gs-tok.noncode{color:var(--code);}
.gs-pinnum{display:inline-block;background:var(--pin);color:#fff;font-size:9px;font-weight:700;
  font-style:normal;line-height:1;padding:1px 3px;border-radius:6px;margin-left:1px;vertical-align:super;}
svg.gsf-links{position:absolute;inset:0;width:100%;height:100%;pointer-events:none;z-index:2;overflow:visible;}
.gsf-callout{position:absolute;width:232px;background:#fff;border:1px solid var(--box);
  border-radius:12px;padding:11px 14px 8px;box-shadow:0 1px 3px rgba(16,24,40,.07);z-index:3;
  font-size:14px;line-height:1.42;font-style:italic;color:#111827;}
.gsf-callout.nocontrols{padding-bottom:11px;}  /* no ctrls row to leave room for */
.gsf-callout.side-left{left:20px;max-width:calc(var(--marginLw) - 40px);}
.gsf-callout.side-right{right:20px;max-width:calc(var(--marginRw) - 40px);}
.gsf-callout .idx{position:absolute;left:-11px;top:-11px;width:22px;height:22px;border-radius:50%;
  background:var(--pin);color:#fff;font-style:normal;font-weight:700;font-size:12px;
  display:flex;align-items:center;justify-content:center;box-shadow:0 1px 2px rgba(0,0,0,.2);}
.gsf-callout .body{display:block;}
.gsf-callout .ctrls{display:flex;gap:3px;margin-top:7px;padding-top:6px;border-top:1px solid #f1f2f4;
  font-style:normal;align-items:center;}
.gsf-callout .ctrls button{border:1px solid var(--box);background:#fafafa;border-radius:6px;
  width:22px;height:20px;font-size:11px;line-height:1;cursor:pointer;color:var(--muted);padding:0;}
.gsf-callout .ctrls button:hover{background:#eef0f2;color:#111827;}
.gsf-callout .ctrls button.active{background:var(--pin);color:#fff;border-color:var(--pin);}
.gsf-callout .ctrls .lbl{font-size:10px;color:var(--muted);letter-spacing:.02em;text-transform:uppercase;margin-right:auto;}
.gsf-hint{font-size:12px;color:var(--muted);font-style:italic;margin-bottom:14px;}
.gsf-tip{position:fixed;pointer-events:none;z-index:9;max-width:320px;background:#111827;color:#fff;
  font-size:13px;line-height:1.4;font-style:italic;padding:8px 11px;border-radius:8px;opacity:0;
  transition:opacity .1s;box-shadow:0 4px 14px rgba(0,0,0,.25);}
""".strip()

# {CID} substituted with a per-render container id.
_JS_FLOW = r"""
(function(){
  const CID = "__CID__";
  const root = document.getElementById(CID);
  if(!root) return;
  const TOK = __TOKENS_JSON__;                 // [{pos,text,readout,scoped}]
  const SHOW_CONTROLS = __SHOW_CONTROLS__;      // false hides the move (◀▶▼) row -- for print/figure export
  const readouts = {}; TOK.forEach(t=>{ if(t.scoped) readouts[t.pos]=t.readout; });
  const textEl = root.querySelector(".gsf-text");
  const marginL = root.querySelector(".gsf-marginL");
  const marginR = root.querySelector(".gsf-marginR");
  const marginB = root.querySelector(".gsf-marginB");
  const svg    = root.querySelector("svg.gsf-links");
  const tip    = root.querySelector(".gsf-tip");
  const pinned = [];                           // ordered list of pos (pin order)
  const place  = {};                            // pos -> "left"|"right"|"below"
  const CW = 232, GAP = 12;

  function esc(s){ return (s||"").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }

  root.querySelectorAll(".gs-tok.scoped").forEach(el=>{
    el.addEventListener("mousemove", e=>{
      const p = +el.dataset.pos;
      tip.textContent = "\u201c"+ (readouts[p]||"") +"\u201d";
      tip.style.opacity = .98;
      tip.style.left = Math.min(e.clientX+14, window.innerWidth-336) + "px";
      tip.style.top  = (e.clientY+16) + "px";
    });
    el.addEventListener("mouseleave", ()=>{ tip.style.opacity = 0; });
    el.addEventListener("click", ()=>{
      const p = +el.dataset.pos;
      const i = pinned.indexOf(p);
      if(i>=0){ pinned.splice(i,1); el.classList.remove("pinned"); delete place[p]; }
      else    {
        pinned.push(p); el.classList.add("pinned");
        // default: balance -- put it in whichever margin currently holds fewer pins
        // ("below" is opt-in only, via the button -- never the default)
        const nL = pinned.filter(q=>place[q]==="left").length;
        const nR = pinned.filter(q=>place[q]==="right").length;
        place[p] = nL <= nR ? "left" : "right";
      }
      render();
    });
  });

  function calloutInner(n, p, s){
    let html = '<span class="idx">'+n+'</span>'
      +'<span class="body">\u201c'+esc(readouts[p])+'\u201d</span>';
    if(SHOW_CONTROLS){
      html += '<div class="ctrls">'
        +'<span class="lbl">move</span>'
        +'<button data-place="left" title="left margin">\u25c0</button>'
        +'<button data-place="right" title="right margin">\u25b6</button>'
        +'<button data-place="below" title="bottom margin">\u25bc</button>'
      +'</div>';
    }
    return html;
  }

  function render(){
    root.querySelectorAll(".gsf-callout").forEach(n=>n.remove());
    root.querySelectorAll(".gs-pinnum").forEach(n=>n.remove());
    while(svg.firstChild) svg.removeChild(svg.firstChild);

    // small numbered badge right on the token -- matches its callout's own number, so which
    // callout belongs to which token stays obvious even once a few leader lines cross.
    pinned.forEach((p,i)=>{
      const t = textEl.querySelector('.gs-tok[data-pos="'+p+'"]');
      const badge = document.createElement("sup");
      badge.className = "gs-pinnum";
      badge.textContent = i+1;
      t.insertAdjacentElement("afterend", badge);
    });

    let hint = root.querySelector(".gsf-hint");
    if(pinned.length===0){
      if(hint) hint.textContent = "Click a highlighted token below to pin its read-out.";
      marginB.style.minHeight = "1px";
      return;
    }
    if(hint) hint.textContent = SHOW_CONTROLS
      ? pinned.length + " pinned \u2014 use \u25c0\u25b6\u25bc on a callout to move it."
      : pinned.length + " pinned.";

    const O = svg.getBoundingClientRect();

    // "left"/"right" -- absolutely positioned in the reserved side margins, stacked top-to-bottom
    // by the token's vertical position, connected by a leader line to the token's NEAR edge (not
    // its center, so neither the line nor the box ever sits on top of the word).
    ["left","right"].forEach(s=>{
      const col = s==="left" ? marginL : marginR;
      const order = pinned.filter(p=>place[p]===s).map(p=>{
        const t = textEl.querySelector('.gs-tok[data-pos="'+p+'"]');
        const r = t.getBoundingClientRect();
        return {p, t, ty:(r.top+r.bottom)/2 - O.top};
      }).sort((a,b)=>a.ty-b.ty);

      let lastBottom = 0;
      order.forEach(o=>{
        const box = document.createElement("div");
        box.className = "gsf-callout side-" + s + (SHOW_CONTROLS ? "" : " nocontrols");
        const n = pinned.indexOf(o.p) + 1;
        box.innerHTML = calloutInner(n, o.p, s);
        col.appendChild(box);
        box.querySelectorAll("button").forEach(btn=>{
          if(btn.dataset.place === s) btn.classList.add("active");
          btn.addEventListener("click", ()=>{ place[o.p] = btn.dataset.place; render(); });
        });
        const bh = box.offsetHeight;
        let top = Math.max(o.ty - bh/2, lastBottom + GAP);
        box.style.top = top + "px";
        lastBottom = top + bh;
        o.box = box;
      });

      order.forEach(o=>{
        const br = o.box.getBoundingClientRect();
        const x1 = s==="left" ? br.right - O.left : br.left - O.left;
        const y1 = br.top - O.top + o.box.offsetHeight/2;
        const tr = o.t.getBoundingClientRect();
        const x2 = (s==="left" ? tr.left : tr.right) - O.left, y2 = (tr.top+tr.bottom)/2 - O.top;
        const mx = (x1+x2)/2;
        const path = document.createElementNS("http://www.w3.org/2000/svg","path");
        path.setAttribute("d", `M ${x1} ${y1} C ${mx} ${y1}, ${mx} ${y2}, ${x2} ${y2}`);
        path.setAttribute("fill","none"); path.setAttribute("stroke","#9aa1ab");
        path.setAttribute("stroke-width","1.4");
        svg.appendChild(path);
      });
    });

    // "below" -- a reserved full-width strip under the whole text column (mirrors the left/right
    // margins), packed strictly left-to-right (sorted by each token's own x-position, so reading
    // order matches box order) and wrapping to a new row when a shelf fills up. Packing always
    // continues from the previous box's right edge -- never jumps ahead to "under my own token" --
    // so there's never a gap one box won't fill or an overlap from being placed out of order; the
    // leader line is what shows which token a box belongs to, not its exact horizontal position.
    const belowOrder = pinned.filter(p=>place[p]==="below").map(p=>{
      const t = textEl.querySelector('.gs-tok[data-pos="'+p+'"]');
      const tr = t.getBoundingClientRect();
      return {p, t, tx:(tr.left+tr.right)/2};
    }).sort((a,b)=>a.tx-b.tx);
    const BR = marginB.getBoundingClientRect();
    let rowTop = 0, rowH = 0, cursorX = 0;
    belowOrder.forEach(o=>{
      const box = document.createElement("div");
      box.className = "gsf-callout" + (SHOW_CONTROLS ? "" : " nocontrols");
      const n = pinned.indexOf(o.p) + 1;
      box.innerHTML = calloutInner(n, o.p, "below");
      marginB.appendChild(box);
      box.querySelectorAll("button").forEach(btn=>{
        if(btn.dataset.place === "below") btn.classList.add("active");
        btn.addEventListener("click", ()=>{ place[o.p] = btn.dataset.place; render(); });
      });

      let left = cursorX;
      if(left + CW > BR.width){ rowTop += rowH + GAP; rowH = 0; left = 0; }
      box.style.left = left + "px";
      box.style.top  = rowTop + "px";
      const bh = box.offsetHeight;
      rowH = Math.max(rowH, bh);
      cursorX = left + CW + GAP;

      const tr = o.t.getBoundingClientRect();
      const x1 = (tr.left+tr.right)/2 - O.left, y1 = tr.bottom - O.top;
      const x2 = (BR.left - O.left) + left + CW/2, y2 = (BR.top - O.top) + rowTop;
      const my = (y1+y2)/2;
      const path = document.createElementNS("http://www.w3.org/2000/svg","path");
      path.setAttribute("d", `M ${x1} ${y1} C ${x1} ${my}, ${x2} ${my}, ${x2} ${y2}`);
      path.setAttribute("fill","none"); path.setAttribute("stroke","#9aa1ab");
      path.setAttribute("stroke-width","1.4");
      svg.appendChild(path);
    });
    marginB.style.minHeight = belowOrder.length ? (rowTop + rowH + 4) + "px" : "1px";
  }
  window.addEventListener("resize", render);

  // draggable margin dividers -- resize the left/right margin columns; callouts + leader lines
  // re-flow live. Bottom margin isn't draggable: its height auto-fits its own content, so there's
  // nothing meaningful to drag there.
  let dragging = null;   // "left" | "right" | null
  root.querySelectorAll(".gsf-div").forEach(div=>{
    div.addEventListener("mousedown", e=>{
      dragging = div.dataset.side; div.classList.add("drag");
      document.body.style.userSelect = "none"; e.preventDefault();
    });
  });
  document.addEventListener("mousemove", e=>{
    if(!dragging) return;
    const wr = root.getBoundingClientRect();
    const w = dragging === "left" ? (e.clientX - wr.left) : (wr.right - e.clientX);
    const clamped = Math.min(420, Math.max(160, w));
    root.style.setProperty(dragging === "left" ? "--marginLw" : "--marginRw", clamped + "px");
    render();
  });
  document.addEventListener("mouseup", ()=>{
    if(dragging){
      root.querySelectorAll(".gsf-div.drag").forEach(d=>d.classList.remove("drag"));
      dragging = null; document.body.style.userSelect=""; render();
    }
  });

  render();
})();
""".strip()

def goalscope_qualitative_flow_html(
    results: list[dict],
    *,
    title: str = "GoalScope",
    show_controls: bool = True,
    save_path: Optional[str] = None,
    container_id: Optional[str] = None,
) -> str:
    """Like :func:`goalscope_qualitative_html`, but for natural (non-code) text: each callout
    snaps to one of three RESERVED margins -- LEFT, RIGHT (two fixed columns flanking a narrower
    text column) or BOTTOM (a full-width strip below the whole text, shelf-packed under each
    token's own x-position). A fixed left column reads fine for code (one clear "reading
    direction"), but for prose the relevant token can be anywhere in the paragraph -- a
    single-side column either draws very long leader lines for tokens on the far side, or crowds
    one edge.

    Each pinned callout defaults to whichever side margin currently holds fewer pins (a simple
    balance heuristic; "below" is opt-in only, never the default) and carries ◀▶▼ buttons to
    move it. This is NOT free-drag positioning -- all three margins are reserved space, so a
    callout can never cover the text itself, which is the main failure mode free-floating
    placement ran into; full drag-anywhere would need its own collision/leader-line bookkeeping on
    every mousemove and wasn't worth the added complexity (and risk, un-clicktested) for what
    three reserved margins already solve.

    Args:
        results: list of dicts with ``pos`` / ``token`` / ``patched_gen`` / ``scoped``, i.e. the
            output of ``apply_goalscope(..., include_noncode=True, show=False)``.
        title: shown above the text panel.
        show_controls: when False, hides the ◀▶▼ "move" row on every callout (and
            simplifies the pin-count hint) -- for a static figure (e.g. printed to PDF for a
            paper) where the interactive controls have no purpose. Positions still default to the
            same left/right balance heuristic; you just can't override them without controls.
        save_path: if given, write the standalone HTML doc there.
        container_id: DOM id (auto-generated if None).

    Returns:
        The standalone HTML document string.
    """
    cid = container_id or ("gsf_" + uuid.uuid4().hex[:8])
    rows = sorted(results, key=lambda r: r["pos"])

    spans = []
    tokens_json = []
    for r in rows:
        pos = int(r["pos"])
        text = r.get("token", "")
        scoped = bool(r.get("scoped", True)) and bool((r.get("patched_gen") or "").strip())
        cls = ["gs-tok", "scoped" if scoped else "noncode"]
        tok_html = _html.escape(text).replace("\n", "↵\n")
        spans.append(f'<span class="{" ".join(cls)}" data-pos="{pos}">{tok_html}</span>')
        tokens_json.append(
            {"pos": pos, "text": text, "scoped": scoped,
             "readout": (r.get("patched_gen") or "").strip() if scoped else ""}
        )

    text_html = "".join(spans)
    js = _JS_FLOW.replace("__CID__", cid).replace(
        "__TOKENS_JSON__", json.dumps(tokens_json, ensure_ascii=False)
    ).replace("__SHOW_CONTROLS__", "true" if show_controls else "false")
    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_html.escape(title)}</title><style>{_CSS_FLOW}</style></head>
<body>
<div class="gsf-wrap" id="{cid}">
  <div class="gsf-row">
    <div class="gsf-marginL"></div>
    <div class="gsf-div" data-side="left" title="drag to resize"></div>
    <div class="gsf-textcol">
      <div class="gsf-title">{_html.escape(title)}</div>
      <div class="gsf-hint">Click a highlighted token below to pin its read-out.</div>
      <div class="gsf-text">{text_html}</div>
    </div>
    <div class="gsf-div" data-side="right" title="drag to resize"></div>
    <div class="gsf-marginR"></div>
  </div>
  <div class="gsf-marginB"></div>
  <svg class="gsf-links"></svg>
  <div class="gsf-tip"></div>
</div>
<script>{js}</script>
</body></html>"""

    if save_path:
        with open(save_path, "w", encoding="utf-8") as fh:
            fh.write(doc)
    return doc


def show_goalscope_qualitative_flow(results: list[dict], *, height: int = 760, **kwargs):
    """Inline (iframe) render of :func:`goalscope_qualitative_flow_html`."""
    from IPython.display import HTML

    doc = goalscope_qualitative_flow_html(results, **kwargs)
    srcdoc = _html.escape(doc, quote=True)
    return HTML(
        f'<iframe srcdoc="{srcdoc}" style="width:100%;height:{height}px;border:1px solid #e5e7eb;'
        f'border-radius:10px;" sandbox="allow-scripts"></iframe>'
    )


def show_goalscope_qualitative(results: list[dict], *, height: int = 760, **kwargs):
    """Return an IPython ``HTML`` object that renders the demo inline via a sandboxed ``iframe``
    (``srcdoc``) so the interactive JS runs reliably across notebook frontends."""
    from IPython.display import HTML

    doc = goalscope_qualitative_html(results, **kwargs)
    srcdoc = _html.escape(doc, quote=True)
    return HTML(
        f'<iframe srcdoc="{srcdoc}" style="width:100%;height:{height}px;border:1px solid #e5e7eb;'
        f'border-radius:10px;" sandbox="allow-scripts"></iframe>'
    )
