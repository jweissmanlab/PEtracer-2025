import sys
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics

# Configure paths
figure_path = Path(__file__).parent
base_path = Path(__file__).parent.parent
preedited_path = base_path / "preedited" / "plots"
sys.path.append(str(base_path))

# Load source
from petracer.utils import render_plot

# Make canvas
c = canvas.Canvas(str(figure_path / "s8.pdf"), pagesize=(8.5*72, 11*72))
pdfmetrics.registerFont(TTFont('Arial-Bold', 'Arial_Bold.ttf'))
c.setFont('Arial-Bold', 14)

# Render panels
render_plot(c, "A", preedited_path / "merfish_invivo_slide.svg", 0, 0, x_offset=25,y_offset=10)
render_plot(c, "", preedited_path / "merfish_invivo_fov.svg", 3.5, 0, x_offset=30,y_offset=10)
render_plot(c, "B", preedited_path / "merfish_invivo_characters.svg", 0, 3.6, x_offset=30,y_offset=10)
render_plot(c, "C", preedited_path / "merfish_invivo_integration_confusion_matrix.svg", 4.2, 3.6, x_offset=30,y_offset=15)
render_plot(c, "D", preedited_path / "merfish_invivo_detection_stats_barplot.svg", 4.2, 4.7, x_offset=25)
render_plot(c, "E", preedited_path / "merfish_invivo_edit_confusion_matrix.svg", 0, 6.6, x_offset=30,y_offset=15)



# Save canvas
c.save()