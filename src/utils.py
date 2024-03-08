from pathlib import Path
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF


def save_plot(plt, plot_name, plots_path='.'):
    plots_path = Path(plots_path)
    for fmt in ['png', 'svg']:
        plt.savefig(plots_path / f"{plot_name}.{fmt}", bbox_inches='tight')


def render_plot(c, label, path, x, y, scale = 1,x_offset=10,y_offset=20):
    x = x * 72
    y = (11 - y) * 72
    if path is not None:
        drawing = svg2rlg(str(path))
        drawing.scale(scale, scale)
        renderPDF.draw(drawing, c,x + x_offset,
                       y-drawing.height * scale - y_offset)
    if label is not None:
        c.drawString(x+10, y - 20, label)