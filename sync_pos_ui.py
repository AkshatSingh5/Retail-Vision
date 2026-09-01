from pathlib import Path
import shutil

root = Path(__file__).resolve().parent
src_html = (root / "frontend" / "index.html").read_text(encoding="utf-8")
src_html = src_html.replace("/src/favicon.png", "/static/pos/favicon.png")
src_html = src_html.replace("/src/pos.css", "/static/pos/pos.css")
src_html = src_html.replace("/src/config.js", "/static/pos/config.js")
src_html = src_html.replace("/src/api.js", "/static/pos/api.js")
src_html = src_html.replace("/src/pos.js", "/static/pos/pos.js")
dest = root / "backend" / "app" / "static" / "pos"
dest.mkdir(parents=True, exist_ok=True)
(dest / "index.html").write_text(src_html, encoding="utf-8")
shutil.copy2(root / "frontend" / "src" / "pos.css", dest / "pos.css")
shutil.copy2(root / "frontend" / "src" / "config.js", dest / "config.js")
shutil.copy2(root / "frontend" / "src" / "api.js", dest / "api.js")
shutil.copy2(root / "frontend" / "src" / "pos.js", dest / "pos.js")
print("copied", dest)
print("PRODUCT SCANNER", "PRODUCT SCANNER" in src_html)
print("config leftover", "/src/config.js" in src_html)
