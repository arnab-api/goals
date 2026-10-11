/* Nav sidebar behaviour shared by index.html, icl.html and em.html.
 *
 * Desktop: drag the sidebar's left edge to widen it, from its default width (--sidebar-w in site.css) up to
 * min(1.5x default, 30% of the window); never narrower than the default. Double-click the edge to reset. The width
 * is remembered per browser (localStorage) and re-clamped when the window is resized.
 *
 * Phone (<= 760px, see site.css): the sidebar becomes a bar on top; its task list opens with the "Tasks" toggle and
 * closes again once a task is picked. The toggle shows the selected task.
 */
(function () {
  const sb = document.getElementById("sidebar");
  if (!sb) return;
  const root = document.documentElement, KEY = "gs-sidebar-w";
  const base = parseFloat(getComputedStyle(root).getPropertyValue("--sidebar-w")) || 240;
  const maxW = () => Math.max(base, Math.min(base * 1.5, innerWidth * 0.3));
  let want = null;                                   // the user's chosen width (px), before clamping
  const apply = () => {
    if (want == null) root.style.removeProperty("--sidebar-w");
    else root.style.setProperty("--sidebar-w", Math.min(maxW(), Math.max(base, want)) + "px");
  };
  try { const w = parseFloat(localStorage.getItem(KEY)); if (w) { want = w; apply(); } } catch (e) {}
  addEventListener("resize", e => { if (e.isTrusted) apply(); });

  // ── desktop: resize handle ──
  const h = document.createElement("div");
  h.className = "sb-resize"; h.title = "drag to resize the menu (double-click to reset)";
  sb.appendChild(h);
  h.addEventListener("pointerdown", e => {
    if (e.button !== 0) return;
    h.setPointerCapture(e.pointerId); h.classList.add("drag"); document.body.classList.add("sb-resizing"); e.preventDefault();
    const right = sb.getBoundingClientRect().right;
    const move = ev => { want = right - ev.clientX; apply(); dispatchEvent(new Event("resize")); };
    const up = () => {
      h.removeEventListener("pointermove", move); h.removeEventListener("pointerup", up);
      h.classList.remove("drag"); document.body.classList.remove("sb-resizing");
      want = parseFloat(getComputedStyle(root).getPropertyValue("--sidebar-w")) || base;
      try { localStorage.setItem(KEY, want); } catch (e) {}
    };
    h.addEventListener("pointermove", move); h.addEventListener("pointerup", up);
  });
  h.addEventListener("dblclick", () => {
    want = null; apply(); dispatchEvent(new Event("resize"));
    try { localStorage.removeItem(KEY); } catch (e) {}
  });

  // ── phone: collapsible task list ──
  const head = sb.querySelector(".sb-header"), list = document.getElementById("task-list");
  const t = document.createElement("button");
  t.type = "button"; t.className = "sb-toggle"; t.setAttribute("aria-expanded", "false");
  t.innerHTML = '<span class="sb-toggle-label">Tasks</span><span class="sb-toggle-caret" aria-hidden="true">▾</span>';
  head.insertBefore(t, head.querySelector(".sb-nav"));
  const label = t.querySelector(".sb-toggle-label");
  const setOpen = o => { sb.classList.toggle("sb-open", o); t.setAttribute("aria-expanded", String(o)); };
  t.addEventListener("click", () => setOpen(!sb.classList.contains("sb-open")));
  const showSel = () => {
    const li = list && list.querySelector("li.selected");
    if (!li) return;
    const c = li.cloneNode(true); c.querySelectorAll(".acc").forEach(n => n.remove());
    label.textContent = c.textContent.trim() || "Tasks";
  };
  if (list) {
    new MutationObserver(showSel).observe(list, {subtree: true, childList: true, attributes: true, attributeFilter: ["class"]});
    list.addEventListener("click", e => { const li = e.target.closest("li"); if (li && !li.classList.contains("grp")) setOpen(false); });
  }
})();
