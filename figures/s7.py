import sys
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics

# Configure paths
figure_path = Path(__file__).parent
base_path = Path(__file__).parent.parent
barcoding_path = base_path / "barcoding" / "plots"
sys.path.append(str(base_path))

# Load source
from petracer.utils import render_plot

# Make canvas
c = canvas.Canvas(str(figure_path / "s7.pdf"), pagesize=(8.5*72, 11*72))
pdfmetrics.registerFont(TTFont('Arial-Bold', 'Arial_Bold.ttf'))
c.setFont('Arial-Bold', 14)

# Render panels
render_plot(c, "A", barcoding_path / "clone_1_with_barcodes.png", 0, 0,y_offset=10,x_offset=30)
render_plot(c, "", barcoding_path / "clone_2_with_barcodes.png", 0, 1.8,y_offset=10,x_offset=30)
render_plot(c, "", barcoding_path / "clone_3_with_barcodes.png", 0, 3.6,y_offset=10,x_offset=30)
render_plot(c, "", barcoding_path / "clone_4_with_barcodes.png", 0, 5.4,y_offset=10,x_offset=30)
render_plot(c, "", barcoding_path / "clone_5_with_barcodes.png", 0, 7.2,y_offset=10,x_offset=30)
render_plot(c, "", barcoding_path / "clone_6_with_barcodes.png", 0, 9,y_offset=10,x_offset=30)


# Save canvas
c.save()