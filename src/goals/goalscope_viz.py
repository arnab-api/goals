"""Interactive qualitative GoalScope demo (paper-style callouts).

Turns the output of :func:`src.goals.goalscope.apply_goalscope` — a ``list[dict]`` with keys
``pos`` / ``token`` / ``patched_gen`` / ``scoped`` — into a self-contained HTML page in the style of
``Overleaf-draft/REF/qualitative_demo.png``: the source code on the right (syntax-tinted, monospace),
and, on the left, italic-quote callout boxes connected by curved leader lines to pink anchor dots on
the chosen tokens.

Interaction: **click a scoped token** to pin its read-out — a numbered callout appears on the left,
the token is boxed and gets an anchor dot, and a leader line is drawn. Click again to unpin. Multiple
tokens can be pinned at once. Hovering a scoped token shows a transient preview. Drag a callout up/down to
move it (its leader line follows); double-click it to return it to its automatic place.

Usage::

    from src.goals.goalscope import apply_goalscope
    from src.goals.goalscope_viz import goalscope_qualitative_html
    res = apply_goalscope(full_text, mt=mt, selected_heads=heads, include_noncode=True)
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
.gs-callout{cursor:grab;user-select:none;}
.gs-callout.dragging{cursor:grabbing;box-shadow:0 6px 18px rgba(16,24,40,.16);z-index:4;}
.gs-callout .idx{position:absolute;left:-11px;top:-11px;width:22px;height:22px;border-radius:50%;
  background:var(--pin);color:#fff;font-style:normal;font-weight:700;font-size:12px;
  display:flex;align-items:center;justify-content:center;box-shadow:0 1px 2px rgba(0,0,0,.2);}
.gs-hint{position:absolute;top:22px;right:26px;font-size:12px;color:var(--muted);font-style:italic;}
.gs-tip{position:fixed;pointer-events:none;z-index:9;max-width:320px;background:#111827;color:#fff;
  font-size:13px;line-height:1.4;font-style:italic;padding:8px 11px;border-radius:8px;opacity:0;
  transition:opacity .1s;box-shadow:0 4px 14px rgba(0,0,0,.25);}
.gs-empty{color:var(--muted);font-size:14px;font-style:italic;padding-top:6px;}
/* stacked view when the PAGE/panel is narrow (render() sets .narrow from the wrap's own width): callouts below the
   code, a pinned token is prefixed with its callout number */
.gs-wrap.narrow{flex-direction:column;padding:14px 12px 28px;}
.gs-wrap.narrow .gs-left{flex:none;order:2;min-height:0;}
.gs-wrap.narrow .gs-right{padding:0;}
.gs-wrap.narrow .gs-divider,.gs-wrap.narrow svg.gs-links{display:none;}
.gs-wrap.narrow pre.gs-code{font-size:13px;}
.gs-wrap.narrow .gs-callout{position:relative;left:auto;right:auto;top:auto!important;max-width:none;margin:14px 0 0 10px;cursor:default;}
.gs-wrap.narrow .gs-tok.pinned[data-n]::before{content:attr(data-n);display:inline-block;margin-right:2px;padding:0 4px;
  border-radius:7px;background:var(--pin);color:#fff;font:700 9.5px/14px ui-sans-serif,system-ui,sans-serif;
  font-style:normal;vertical-align:super;}
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
  const manual = {};                           // pos -> callout top relative to its token's centre (px), after a drag
  const tops = {}, tys = {};                   // pos -> current callout top / token centre (px)
  const NARROW = 440;                          // wrap width (px) below which callouts stack under the code
  let dragPos = null;                          // pos of the callout being dragged

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
      if(i>=0){ pinned.splice(i,1); delete manual[p]; el.classList.remove("pinned"); }
      else    { pinned.push(p);     el.classList.add("pinned"); }
      render();
    });
  });

  // vertical drag of a callout; render() re-creates the boxes, so the drag is tracked by token position
  function makeDraggable(box, p){
    if(dragPos === p) box.classList.add("dragging");
    box.addEventListener("dblclick", ()=>{ delete manual[p]; render(); });
    box.addEventListener("pointerdown", e=>{
      if(e.button !== 0) return;
      const y0 = e.clientY, t0 = parseFloat(box.style.top) || 0;
      let moved = false;
      const move = ev=>{
        const dy = ev.clientY - y0; if(!moved && Math.abs(dy) < 3) return;
        if(!moved) pinned.forEach(q=>{ if(!(q in manual) && q in tops) manual[q] = tops[q] - tys[q]; });  // freeze the others
        moved = true; manual[p] = Math.max(0, t0 + dy) - tys[p]; dragPos = p; render();
      };
      const up = ()=>{ window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up);
                       dragPos = null; if(moved) render(); };
      window.addEventListener("pointermove", move); window.addEventListener("pointerup", up); e.preventDefault();
    });
  }

  function render(){
    // clear callouts + links
    root.querySelectorAll(".gs-callout").forEach(n=>n.remove());
    while(svg.firstChild) svg.removeChild(svg.firstChild);
    const empty = left.querySelector(".gs-empty"); if(empty) empty.remove();
    right.querySelectorAll(".gs-tok").forEach(t=>{
      const i = pinned.indexOf(+t.dataset.pos); if(i>=0) t.dataset.n = i+1; else delete t.dataset.n;
    });
    const narrow = root.clientWidth < NARROW;
    root.classList.toggle("narrow", narrow);
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
      if(narrow){ left.appendChild(box); return; }       // stacked view: no positioning, no leader lines
      root.appendChild(box);
      const bh = box.offsetHeight;
      let top = (o.p in manual) ? Math.max(0, o.ty + manual[o.p]) : Math.max(o.ty - bh/2, lastBottom + 12);
      box.style.top = top + "px";
      lastBottom = Math.max(lastBottom, top + bh); tops[o.p] = top; tys[o.p] = o.ty;
      o.box = box; o.n = n;
      makeDraggable(box, o.p);
    });
    if(narrow) return;
    // draw leader lines + anchor dots (after layout is known)
    order.forEach(o=>{
      const br = o.box.getBoundingClientRect();
      const x1 = br.right - O.left, y1 = br.top - O.top + o.box.offsetHeight/2;
      const tr = o.t.getBoundingClientRect();
      const x2 = tr.left - 2 - O.left, y2 = (tr.top+tr.bottom)/2 - O.top;
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

  // read-outs pinned when the page opens (initial_pins); still unpinnable by click
  __INITIAL_PINS__.forEach(p=>{
    const el = right.querySelector('.gs-tok.scoped[data-pos="'+p+'"]');
    if(el && pinned.indexOf(p)<0){ pinned.push(p); el.classList.add("pinned"); }
  });
  render();
})();
""".strip()


def goalscope_qualitative_html(
    results: list[dict],
    *,
    title: str = "GoalScope",
    save_path: Optional[str] = None,
    initial_pins: Optional[list[int]] = None,
) -> str:
    """Build a self-contained standalone HTML document from ``apply_goalscope`` output.

    Args:
        results: list of dicts with ``pos`` / ``token`` / ``patched_gen`` / ``scoped``. Pass the
            output of ``apply_goalscope(..., include_noncode=True)`` so the full source
            prompt is shown (non-code tokens are dimmed and not clickable).
        title: shown above the code panel.
        save_path: if given, write the standalone HTML doc there.
        initial_pins: token positions whose read-outs are pinned when the page opens, numbered in this
            order (positions that are not scoped tokens are ignored).

    Returns:
        The standalone HTML document string.
    """
    cid = "gs_" + uuid.uuid4().hex[:8]
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
        if bool(r.get("scoped", True)):   # tint code only — prompt text stays uniformly dim
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
    ).replace("__INITIAL_PINS__", json.dumps([int(p) for p in (initial_pins or [])]))
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

