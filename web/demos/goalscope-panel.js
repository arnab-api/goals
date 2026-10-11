/* GoalScope read-out panel — the paper-style callout view of src/goals/goalscope_viz.py, as a reusable component.
 *
 *   const p = new GoalScopePanel(el, {rows, title, continuation, badge, onToggle});
 *   p.setPins([60, 70]);  p.toggle(78);  p.focus(70);  p.destroy();
 *
 * rows: [{pos, token, patched_gen, scoped, verdict?, tone?}]  (web/demos/demo_data.py)
 * Read-outs sit LEFT of the text (they come before the text they anticipate); click a highlighted token to pin
 * its read-out, click again to unpin, hover to preview. Drag a callout up/down to move it (its leader line follows);
 * double-click it to return it to its automatic place. Drag the vertical divider to resize the two sides. Prompt tokens are dimmed (dimContext); `continuation` (the
 * model's own generation) follows in normal ink. A read-out with tone "concerning" marks its token red.
 */
(function () {
  const CSS = `
.gsp{position:relative;display:flex;gap:0;--leftw:44%;font-family:inherit}
.gsp-left{position:relative;flex:0 0 var(--leftw);min-height:60px}
.gsp-right{flex:1 1 auto;min-width:0;padding-left:18px}
.gsp-divider{flex:0 0 14px;align-self:stretch;position:relative;cursor:col-resize;z-index:4;touch-action:none}
.gsp-divider::before{content:"";position:absolute;left:50%;top:0;bottom:0;width:1px;background:#e5e7eb;transform:translateX(-50%);transition:background .1s,width .1s}
.gsp-divider:hover::before,.gsp-divider.drag::before{background:#9aa1ab;width:2px}
.gsp-title{font:600 11px/1.4 ui-sans-serif,system-ui,sans-serif;letter-spacing:.06em;text-transform:uppercase;color:#6b7280;margin:0 0 8px}
.gsp pre{margin:0;font:13.5px/1.75 ui-monospace,SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace;white-space:pre-wrap;word-break:break-word;color:#1f2328}
.gsp-tok{border-radius:4px;padding:0 .5px}
.gsp-tok.dim{color:#a3a9b3}
.gsp-tok.special{color:#c4c8cf}
.gsp-tok.scoped{white-space:pre;cursor:pointer;background:#fff4c7;box-shadow:inset 0 -1.5px 0 #f2b8c4;transition:background .12s}
.gsp-tok.scoped:hover{background:#ffe9a3}
.gsp-tok.pinned{background:#ffe08a;box-shadow:0 0 0 1.5px #f2708a}
.gsp-tok.focus{outline:2px solid #003636;outline-offset:1px}
.gsp-tok.scoped.concerning{background:#fde4da;box-shadow:inset 0 -1.5px 0 #c2410c;color:#9a3412}
.gsp-tok.scoped.concerning:hover{background:#fbd3c3}
.gsp-tok.pinned.concerning{background:#fbc6b1;box-shadow:0 0 0 1.5px #c2410c}
.gsp-cont{color:#1f2328}
svg.gsp-links{position:absolute;inset:0;width:100%;height:100%;pointer-events:none;overflow:visible;z-index:2}
.gsp-callout{position:absolute;left:14px;width:calc(var(--leftw) - 40px);background:#fff;border:1px solid #d0d5dd;
  border-radius:12px;padding:9px 12px;box-shadow:0 1px 3px rgba(16,24,40,.07);z-index:3;font-size:13.5px;line-height:1.45;color:#111827}
.gsp-callout{cursor:grab;user-select:none}
.gsp-callout.dragging{cursor:grabbing;box-shadow:0 6px 18px rgba(16,24,40,.16);z-index:4}
.gsp-callout q{font-style:italic;quotes:"\\201C" "\\201D"}
.gsp-callout .idx{position:absolute;left:-10px;top:-10px;width:20px;height:20px;border-radius:50%;background:#f2708a;color:#fff;
  font:700 11px/20px ui-sans-serif,system-ui,sans-serif;text-align:center;box-shadow:0 1px 2px rgba(0,0,0,.2)}
.gsp-callout .tag{display:inline-block;margin-top:6px;font:600 10.5px/1 ui-sans-serif,system-ui,sans-serif;letter-spacing:.04em;
  text-transform:uppercase;padding:3px 6px;border-radius:999px;border:1px solid}
.gsp-callout.concerning{border-color:#e7a08a}
.tag.yes{color:#166534;border-color:#86c79a;background:#f0faf3}
.tag.no{color:#9a3412;border-color:#e7a08a;background:#fdf3ef}
.tag.benign{color:#334155;border-color:#b6c2cf;background:#f4f6f9}
.tag.concerning{color:#9a3412;border-color:#e7a08a;background:#fdf3ef}
.gsp-empty{color:#6b7280;font-size:13px;font-style:italic;padding:4px 14px 0 0}
.gsp-tip{position:fixed;pointer-events:none;z-index:99;max-width:320px;background:#111827;color:#fff;font-size:12.5px;line-height:1.4;
  font-style:italic;padding:7px 10px;border-radius:8px;opacity:0;transition:opacity .1s;box-shadow:0 4px 14px rgba(0,0,0,.25)}
/* stacked view when the PANEL is narrow (render() sets .narrow from the panel's own width, not the window's) */
.gsp.narrow{flex-direction:column}.gsp.narrow .gsp-left{flex:none;order:2;min-height:0}
.gsp.narrow .gsp-right{padding-left:0}.gsp.narrow .gsp-divider{display:none}.gsp.narrow svg.gsp-links{display:none}
.gsp.narrow .gsp-callout{position:relative;left:auto;width:auto;top:auto!important;margin:14px 0 0 10px;cursor:default}
.gsp.narrow .gsp-tok.pinned[data-n]::before{content:attr(data-n);display:inline-block;margin-right:2px;padding:0 4px;border-radius:7px;
  background:#f2708a;color:#fff;font:700 9.5px/14px ui-sans-serif,system-ui,sans-serif;vertical-align:super}`;
  if (!document.getElementById("gsp-css")) {
    const s = document.createElement("style"); s.id = "gsp-css"; s.textContent = CSS; document.head.appendChild(s);
  }
  const esc = s => (s || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const SPECIAL = /^<\|[a-z_]+\|>$/;
  const NARROW = 440;                  // panel width (px) below which callouts stack under the text
  let tip = null;

  class GoalScopePanel {
    constructor(el, opts) {
      this.el = el; this.o = Object.assign({badge: null, onToggle: null, dimContext: true}, opts);
      // manual: pos -> callout top relative to its token's centre (px) after a drag, so it follows the token on re-wrap
      this.pins = []; this.ro = {}; this.manual = {}; this._tops = {}; this._ty = {};
      this.o.rows.forEach(r => { if (r.scoped && (r.patched_gen || "").trim()) this.ro[r.pos] = r; });
      if (!tip) { tip = document.createElement("div"); tip.className = "gsp-tip"; document.body.appendChild(tip); }
      const toks = this.o.rows.map(r => {
        const sc = this.ro[r.pos] ? " scoped" + (this.ro[r.pos].tone === "concerning" ? " concerning" : "") : SPECIAL.test(r.token) ? " special" : this.o.dimContext ? " dim" : "";
        return `<span class="gsp-tok${sc}" data-pos="${r.pos}">${esc(r.token).replace(/\n/g, "↵\n")}</span>`;
      }).join("");
      const cont = this.o.continuation ? `<span class="gsp-cont">${esc(this.o.continuation)}</span>` : "";
      el.innerHTML = `<div class="gsp"><div class="gsp-left"></div><div class="gsp-divider" title="drag to resize"></div><div class="gsp-right">
        ${this.o.title ? `<div class="gsp-title">${esc(this.o.title)}</div>` : ""}<pre>${toks}${cont}</pre></div>
        <svg class="gsp-links"></svg></div>`;
      this.root = el.querySelector(".gsp"); this.left = el.querySelector(".gsp-left");
      this.svg = el.querySelector("svg"); this.pre = el.querySelector("pre");
      this.pre.querySelectorAll(".gsp-tok.scoped").forEach(t => {
        const p = +t.dataset.pos;
        t.addEventListener("mousemove", e => {
          tip.textContent = "“" + this.ro[p].patched_gen.trim() + "”"; tip.style.opacity = .97;
          tip.style.left = Math.min(e.clientX + 14, innerWidth - 336) + "px"; tip.style.top = (e.clientY + 16) + "px";
        });
        t.addEventListener("mouseleave", () => { tip.style.opacity = 0; });
        t.addEventListener("click", () => { this.toggle(p); if (this.o.onToggle) this.o.onToggle(p, this.pins.includes(p)); });
      });
      const dv = el.querySelector(".gsp-divider");           // draggable splitter: read-outs | text
      dv.addEventListener("pointerdown", e => {
        if (e.button !== 0) return;
        dv.classList.add("drag"); document.body.style.userSelect = "none"; e.preventDefault();
        const move = ev => {
          const r = this.root.getBoundingClientRect();
          const f = Math.min(0.80, Math.max(0.18, (ev.clientX - r.left) / r.width));
          this.root.style.setProperty("--leftw", (f * 100).toFixed(2) + "%"); this.render();
        };
        const up = () => { removeEventListener("pointermove", move); removeEventListener("pointerup", up);
                           dv.classList.remove("drag"); document.body.style.userSelect = ""; this.render(); };
        addEventListener("pointermove", move); addEventListener("pointerup", up);
      });
      this._rs = () => this.render(); addEventListener("resize", this._rs);
      if (window.ResizeObserver) {      // re-flow on width changes only (render() itself sets the height)
        let w = 0;
        this._ro = new ResizeObserver(es => { const nw = es[0].contentRect.width; if (nw !== w) { w = nw; this.render(); } });
        this._ro.observe(this.root);
      }
      this.render();
    }
    scoped() { return Object.keys(this.ro).map(Number).sort((a, b) => a - b); }
    setPins(ps) { this.pins = ps.filter(p => this.ro[p]); Object.keys(this.manual).forEach(p => { if (!this.pins.includes(+p)) delete this.manual[p]; }); this.render(); }
    toggle(p) {
      if (!this.ro[p]) return;
      const i = this.pins.indexOf(p); if (i >= 0) { this.pins.splice(i, 1); delete this.manual[p]; } else this.pins.push(p); this.render();
    }
    focus(p) {
      this.pre.querySelectorAll(".gsp-tok.focus").forEach(t => t.classList.remove("focus"));
      const t = this.pre.querySelector(`.gsp-tok[data-pos="${p}"]`); if (t) t.classList.add("focus");
    }
    destroy() { removeEventListener("resize", this._rs); if (this._ro) this._ro.disconnect(); this.el.innerHTML = ""; }
    _draggable(b, p) {                    // vertical drag; render() re-creates the boxes, so track by position.
                                          // A drag first freezes every callout where it is, so they move independently
      b.addEventListener("dblclick", () => { delete this.manual[p]; this.render(); });
      b.addEventListener("pointerdown", e => {
        if (e.button !== 0) return;
        const y0 = e.clientY, t0 = parseFloat(b.style.top) || 0;
        let moved = false;
        const move = ev => {
          const dy = ev.clientY - y0; if (!moved && Math.abs(dy) < 3) return;
          if (!moved) this.pins.forEach(q => { if (!(q in this.manual) && q in this._tops) this.manual[q] = this._tops[q] - this._ty[q]; });
          moved = true; this.manual[p] = Math.max(0, t0 + dy) - this._ty[p]; this._dragging = p; this.render();
        };
        const up = () => { removeEventListener("pointermove", move); removeEventListener("pointerup", up);
                           this._dragging = null; if (moved) this.render(); };
        addEventListener("pointermove", move); addEventListener("pointerup", up); e.preventDefault();
      });
      if (this._dragging === p) b.classList.add("dragging");
    }
    render() {
      this.pre.querySelectorAll(".gsp-tok").forEach(t => {
        const i = this.pins.indexOf(+t.dataset.pos);
        t.classList.toggle("pinned", i >= 0); if (i >= 0) t.dataset.n = i + 1; else delete t.dataset.n;
      });
      const narrow = this.el.clientWidth < NARROW;
      this.root.classList.toggle("narrow", narrow);
      if (narrow) this.root.style.minHeight = "";
      this.root.querySelectorAll(".gsp-callout,.gsp-empty").forEach(n => n.remove());
      while (this.svg.firstChild) this.svg.removeChild(this.svg.firstChild);
      if (!this.pins.length) {
        const e = document.createElement("div"); e.className = "gsp-empty";
        e.textContent = "Click a highlighted token to pin its read-out here."; this.left.appendChild(e); return;
      }
      const O = this.svg.getBoundingClientRect();
      const order = this.pins.map(p => {
        const t = this.pre.querySelector(`.gsp-tok[data-pos="${p}"]`), r = t.getBoundingClientRect();
        return {p, t, ty: (r.top + r.bottom) / 2 - O.top};
      }).sort((a, b) => a.ty - b.ty);
      let last = 0;
      order.forEach(o => {
        const r = this.ro[o.p], b = document.createElement("div"), tag = this.o.badge && r[this.o.badge];
        const label = {yes: "judge: correct", no: "judge: incorrect"}[tag] || tag;
        b.className = "gsp-callout" + (r.tone === "concerning" ? " concerning" : "");
        b.innerHTML = `<span class="idx">${this.pins.indexOf(o.p) + 1}</span><q>${esc(r.patched_gen.trim())}</q>` +
          (tag ? `<br><span class="tag ${esc(tag)}">${esc(label)}</span>` : "");
        (narrow ? this.left : this.root).appendChild(b);
        if (narrow) return;
        const top = o.p in this.manual ? Math.max(0, o.ty + this.manual[o.p]) : Math.max(o.ty - b.offsetHeight / 2, last + 10);
        b.style.top = top + "px"; last = Math.max(last, top + b.offsetHeight); o.b = b;
        this._tops[o.p] = top; this._ty[o.p] = o.ty;
        this._draggable(b, o.p);
      });
      if (narrow) return;
      this.root.style.minHeight = last + "px";
      const preL = this.pre.getBoundingClientRect().left;
      order.forEach(o => {
        const br = o.b.getBoundingClientRect(), tr = o.t.getBoundingClientRect();
        const x1 = br.right - O.left, y1 = br.top - O.top + o.b.offsetHeight / 2, ns = "http://www.w3.org/2000/svg";
        const x0 = preL - O.left - 4;                            // just left of the text column
        let d, x2, y2;
        if (tr.left - preL < 10) {                               // token starts its line: come in from the left
          x2 = tr.left - 2 - O.left; y2 = (tr.top + tr.bottom) / 2 - O.top;
          const mx = (x1 + x2) / 2; d = `M ${x1} ${y1} C ${mx} ${y1}, ${mx} ${y2}, ${x2} ${y2}`;
        } else {                                                 // mid-line: run along the gap under the text line
          x2 = (tr.left + tr.right) / 2 - O.left; y2 = tr.bottom - O.top + 1;
          const mx = (x1 + x0) / 2; d = `M ${x1} ${y1} C ${mx} ${y1}, ${mx} ${y2}, ${x0} ${y2} L ${x2} ${y2}`;
        }
        const path = document.createElementNS(ns, "path");
        path.setAttribute("d", d);
        path.setAttribute("fill", "none"); path.setAttribute("stroke", "#9aa1ab"); path.setAttribute("stroke-width", "1.4");
        const c = document.createElementNS(ns, "circle");
        c.setAttribute("cx", x2); c.setAttribute("cy", y2); c.setAttribute("r", "4");
        c.setAttribute("fill", "#f2708a"); c.setAttribute("stroke", "#fff");
        this.svg.append(path, c);
      });
    }
  }
  window.GoalScopePanel = GoalScopePanel;
})();
