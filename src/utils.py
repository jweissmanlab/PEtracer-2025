from pathlib import Path
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF
from reportlab.lib.utils import ImageReader
import re
import pickle
import matplotlib.pyplot as plt

def convert_svg_text(svg_path):
    with open(svg_path, 'r') as f:
        svg_content = f.readlines()
    
    with open(svg_path, 'w') as file:
        for line in svg_content:
            if '<text' in line or '<tspan' in line:
                # Extract the style attribute
                style_match = re.search(r'style="([^"]*)"', line)
                if style_match:
                    style_content = style_match.group(1)
                    font_size_match = re.search(r'\s*(\d+)px', style_content)
                    font_family_match = re.search(r'\'([^\']+)\'', style_content)
                    font_size = font_size_match.group(1) if font_size_match else '10'
                    font_family = font_family_match.group(1) if font_family_match else 'Arial'
                    font_family = font_family.replace('DejaVu Sans', 'Arial')
                    line = line.replace('style=', f'font-family="{font_family}" font-size="{font_size}" style=')
            file.write(line)

def save_plot(fig, plot_name, plots_path='.',transparent=False,svg = True):
    plt.rcParams['svg.fonttype'] = 'none'
    plots_path = Path(plots_path)
    fig.savefig(plots_path / f"{plot_name}.png", bbox_inches='tight', pad_inches=0, transparent=transparent,dpi = 600)
    if svg:
        fig.savefig(plots_path / f"{plot_name}.svg", bbox_inches='tight', pad_inches=0, transparent=transparent)
        convert_svg_text(plots_path / f"{plot_name}.svg")


def render_plot(c, label, path, x, y, scale=1, x_offset=10, y_offset=25):
    x = x * 72
    y = (11 - y) * 72
    if path is not None:
        file_extension = str(path).split('.')[-1].lower()
        if file_extension == 'svg':
            scale = scale * .8
            drawing = svg2rlg(str(path))
            if scale != 1:
                drawing.scale(scale, scale)
            renderPDF.draw(drawing, c, x + x_offset, y - drawing.height * scale - y_offset)
        elif file_extension == 'png':
            image = ImageReader(str(path))
            iw, ih = image.getSize()
            iw = (iw / 8.275) * scale
            ih = (ih / 8.275) * scale
            c.drawImage(image, x + x_offset, y - ih - y_offset, width=iw, height=ih, mask='auto')
    if label is not None:
        c.drawString(x + 10, y - 20, label)