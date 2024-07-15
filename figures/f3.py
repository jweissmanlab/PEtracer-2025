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
from src.utils import render_plot

# Make canvas
c = canvas.Canvas(str(figure_path / "f3.pdf"), pagesize=(8.5*72, 11*72))
pdfmetrics.registerFont(TTFont('Arial-Bold', 'Arial_Bold.ttf'))
c.setFont('Arial-Bold', 14)

# Render panels
render_plot(c, "A", preedited_path / "preedited_schematic.svg", 0, 0,x_offset=10,scale = 1.1)
render_plot(c, "B", preedited_path / "10x_invitro_characters.svg", 2.3, 0, x_offset=25,y_offset=10)
render_plot(c, "C", preedited_path / "clone_edit_whitelist.svg", 6.5, 0, x_offset=25,y_offset=10)
render_plot(c, "D", preedited_path / "10x_invitro_integration_confusion_matrix.svg", 6.5, 1.15, x_offset=30,y_offset=15)
render_plot(c, "E", preedited_path / "10x_invitro_detection_stats_barplot.svg", 6.5, 2.3, x_offset=25)
render_plot(c, "F", preedited_path / "merfish_invitro_slide.svg", 0, 3.2, x_offset=25,y_offset=10)
render_plot(c, "G", preedited_path / "merfish_invitro_fov.svg", 3.75, 3.2, x_offset=25,y_offset=10)
render_plot(c, "H", preedited_path / "merfish_invitro_cell.svg", 6.5, 4, x_offset=25,y_offset=10)
render_plot(c, "I", preedited_path / "merfish_invitro_spot_images.svg", 0, 5.7)
render_plot(c, "J", preedited_path / "merfish_invitro_intensity_vs_probability.svg", 0, 7.7,x_offset=15,y_offset=15)
render_plot(c, "K", preedited_path / "merfish_invitro_characters.svg", 2.3, 7.7, x_offset=25,y_offset=10)
render_plot(c, "L", preedited_path / "merfish_invitro_integration_confusion_matrix.svg", 6.5, 7.7, x_offset=30,y_offset=15)
render_plot(c, "M", preedited_path / "merfish_invitro_detection_stats_barplot.svg", 6.5, 8.9, x_offset=25)

# Save canvas
c.save()