# concat_py.py — положи в D:\diploma_final
from pathlib import Path

ROOT = Path(r"D:\diploma_final")
OUT  = ROOT / "all_python_code.md"
SKIP = {".git", "__pycache__", ".venv", "venv", ".idea", ".vscode", "node_modules", "HELL_ZONE"}

with OUT.open("w", encoding="utf-8") as f:
    f.write("# AI Media Indexer — полный код Python\n\n")
    for py in sorted(ROOT.rglob("*.py")):
        if any(p in SKIP for p in py.parts):
            continue
        rel = py.relative_to(ROOT)
        f.write(f"\n\n## `{rel}`\n\n```python\n")
        try:
            f.write(py.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            f.write(py.read_text(encoding="cp1251", errors="replace"))
        f.write("\n```\n")

print(f"Готово: {OUT}")