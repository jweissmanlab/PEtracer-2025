from pathlib import Path
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF
from reportlab.lib.utils import ImageReader
import pickle


def save_plot(plt, plot_name, plots_path='.'):
    plots_path = Path(plots_path)
    for fmt in ['png', 'svg']:
        plt.savefig(plots_path / f"{plot_name}.{fmt}", bbox_inches='tight', pad_inches=0)


def render_plot(c, label, path, x, y, scale=1, x_offset=10, y_offset=25):
    scale = scale * .72
    x = x * 72
    y = (11 - y) * 72
    if path is not None:
        file_extension = str(path).split('.')[-1].lower()
        if file_extension == 'svg':
            drawing = svg2rlg(str(path))
            drawing.scale(scale, scale)
            renderPDF.draw(drawing, c, x + x_offset, y - drawing.height * scale - y_offset)
        elif file_extension == 'png':
            image = ImageReader(str(path))
            iw, ih = image.getSize()
            iw = (iw / 300) * 88 * scale
            ih = (ih / 300) * 88 * scale
            c.drawImage(image, x + x_offset, y - ih - y_offset, width=iw, height=ih)
    if label is not None:
        c.drawString(x + 10, y - 20, label)