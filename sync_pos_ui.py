from pathlib import Path
import shutil

root = Path(__file__).resolve().parent
src_html = (root / "frontend" / "index.html").read_text(encoding="utf-8")
src_html = src_html.replace("/src/favicon.png", "/static/pos/favicon.png")
src_html = src_html.replace("/src/pos.css", "/static/pos/pos.css")
src_html = src_html.replace("/src/pos.js", "/static/pos/pos.js")
src_html = src_html.replace('  <script src="/src/config.js"></script>\n', "")
src_html = src_html.replace('  <script src="/src/api.js"></script>\n', "")
dest = root / "backend" / "app" / "static" / "pos"
(dest / "index.html").write_text(src_html, encoding="utf-8")
shutil.copy2(root / "frontend" / "src" / "pos.css", dest / "pos.css")
shutil.copy2(root / "frontend" / "src" / "pos.js", dest / "pos.js")
print("copied", dest)
print("PRODUCT SCANNER", "PRODUCT SCANNER" in src_html)
print("config leftover", "/src/config.js" in src_html)
