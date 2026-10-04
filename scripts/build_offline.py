"""Bundle the trained model and web game into a portable offline HTML file."""
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def build(root=ROOT):
    root = Path(root)
    model = json.loads((root / 'artifacts/q_table.json').read_text(encoding='utf-8'))
    model_js = 'window.TICTACTOE_MODEL=' + json.dumps(model, separators=(',', ':')) + ';\n'
    (root / 'artifacts/model.js').write_text(model_js, encoding='utf-8')
    html = (root / 'index.html').read_text(encoding='utf-8')
    css = (root / 'styles.css').read_text(encoding='utf-8')
    app = (root / 'app.js').read_text(encoding='utf-8')
    favicon = (root / 'assets/favicon.svg').read_text(encoding='utf-8')
    import base64
    icon = 'data:image/svg+xml;base64,' + base64.b64encode(favicon.encode()).decode()
    html = html.replace('href="assets/favicon.svg"', 'href="' + icon + '"')
    html = html.replace('<link rel="stylesheet" href="styles.css">', '<style>\n' + css + '\n</style>')
    html = re.sub(r'\s*<script[^>]+src="artifacts/model.js"[^>]*></script>', '', html)
    html = re.sub(r'\s*<script[^>]+src="app.js"[^>]*></script>', '', html)
    html = html.replace('</body>', '<script>\n' + model_js + '\n' + app.replace('</script', '<\\/script') + '\n</script>\n</body>')
    html = html.replace('href="./"', 'href="#play"')
    path = root / 'TicTacToe_Offline.html'
    path.write_text(html, encoding='utf-8')
    print(f'Offline game: {path} ({path.stat().st_size:,} bytes)')
    return path


if __name__ == '__main__':
    build()
