from pathlib import Path
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF


def save_plot(plt, plot_name, plots_path='.'):
    plots_path = Path(plots_path)
    for fmt in ['png', 'svg']:
        plt.savefig(plots_path / f"{plot_name}.{fmt}", bbox_inches='tight')


def render_panel(c, label, path, x, y, scale = 1):
    x = x * 72
    y = (11 - y) * 72
    if path is None:
        c.drawString(x+10, y - 20, label)
        return
    drawing = svg2rlg(str(path))
    drawing.scale(scale, scale)
    renderPDF.draw(drawing, c,x + 10 ,y-drawing.height * scale - 20)
    c.drawString(x+10, y - 20, label)