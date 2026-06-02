// Click a mermaid diagram to open it full-screen; click again (or Esc) to close.
document.addEventListener("click", function (event) {
  const existing = document.querySelector(".mermaid-overlay");
  if (existing) {
    existing.remove();
    return;
  }

  const diagram = event.target.closest(".mermaid");
  if (!diagram) return;

  const svg = diagram.querySelector("svg");
  if (!svg) return;

  const overlay = document.createElement("div");
  overlay.className = "mermaid-overlay";

  const clone = svg.cloneNode(true);
  clone.removeAttribute("style");
  clone.removeAttribute("width");
  clone.removeAttribute("height");
  overlay.appendChild(clone);

  document.body.appendChild(overlay);
});

document.addEventListener("keydown", function (event) {
  if (event.key === "Escape") {
    const overlay = document.querySelector(".mermaid-overlay");
    if (overlay) overlay.remove();
  }
});
