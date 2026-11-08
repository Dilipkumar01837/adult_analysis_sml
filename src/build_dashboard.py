"""Build results/dashboard.html (self-contained, offline) from results/results.json."""
import json
import sys
from pathlib import Path

TEMPLATE = Path(__file__).with_name("dashboard_template.html")


def build(results_dir="results"):
    results_dir = Path(results_dir)
    data = json.loads((results_dir / "results.json").read_text())
    blob = json.dumps(data).replace("</", "<\\/")  # keep JSON safe inside <script>
    html = TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", blob)
    out = results_dir / "dashboard.html"
    out.write_text(html, encoding="utf-8")
    return out


if __name__ == "__main__":
    print("Wrote", build(sys.argv[1] if len(sys.argv) > 1 else "results"))
