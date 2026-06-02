import os
base = r"D:\tarjimon"
extras = ["extra12.css", "extra13.html", "extra14.css", "extra15.html"]

with open(os.path.join(base, "porto.html"), "r", encoding="utf-8") as f:
    content = f.read()

# Add CSS
extra_css = open(os.path.join(base, "extra12.css"), "r", encoding="utf-8").read()
extra_css += open(os.path.join(base, "extra14.css"), "r", encoding="utf-8").read()
content = content.replace("</head>", "<style>\n" + extra_css + "\n</style>\n</head>", 1)

# Add HTML
extra_html = open(os.path.join(base, "extra13.html"), "r", encoding="utf-8").read()
extra_html += open(os.path.join(base, "extra15.html"), "r", encoding="utf-8").read()
content = content.replace("<script>", extra_html + "\n<script>", 1)

# Add stats2 animation and contribution graph
js_addition = """
<script>
// Stats banner 2 animation
const stats2Observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      const el = entry.target;
      const target = parseFloat(el.dataset.stat2);
      let current = 0;
      const isDecimal = target < 100;
      const increment = target / 80;
      const timer = setInterval(() => {
        current += increment;
        if (current >= target) { current = target; clearInterval(timer); }
        el.textContent = isDecimal ? current.toFixed(1) : Math.floor(current).toLocaleString() + (target >= 1000 ? '+' : '+');
      }, 25);
      stats2Observer.unobserve(el);
    }
  });
}, { threshold: 0.5 });
document.querySelectorAll('[data-stat2]').forEach(el => stats2Observer.observe(el));

// Contribution graph
const graph = document.getElementById('contribGraph');
if (graph) {
  for (let i = 0; i < 365; i++) {
    const cell = document.createElement('div');
    cell.className = 'contrib-cell';
    const r = Math.random();
    if (r > 0.85) cell.classList.add('l5');
    else if (r > 0.7) cell.classList.add('l4');
    else if (r > 0.55) cell.classList.add('l3');
    else if (r > 0.4) cell.classList.add('l2');
    else if (r > 0.25) cell.classList.add('l1');
    graph.appendChild(cell);
  }
}
</script>
"""

content = content.replace("</body>", js_addition + "\n</body>")

with open(os.path.join(base, "porto.html"), "w", encoding="utf-8") as f:
    f.write(content)

print(f"FINAL: {os.path.getsize(os.path.join(base, 'porto.html'))} bytes, {content.count(chr(10))} lines")

# cleanup
for f in extras:
    p = os.path.join(base, f)
    if os.path.exists(p):
        os.remove(p)
        print(f"cleaned: {f}")

# also remove build3
for f in ["build3.py"]:
    p = os.path.join(base, f)
    if os.path.exists(p):
        os.remove(p)
