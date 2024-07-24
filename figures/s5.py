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
c = canvas.Canvas(str(figure_path / "s5.pdf"), pagesize=(8.5*72, 11*72))
pdfmetrics.registerFont(TTFont('Arial-Bold', 'Arial_Bold.ttf'))
c.setFont('Arial-Bold', 14)

# Render panels
render_plot(c, "A", preedited_path / "10x_invitro_umi_histplot.svg", 0, 0, x_offset=30,y_offset=15)
render_plot(c, "B", preedited_path / "preedited_barcode_mapping_heatmap.svg", 3, 0, x_offset=30,y_offset=15)
render_plot(c, "C", preedited_path / "merfish_zombie_slide.svg", 0, 2.6, x_offset=10,y_offset=30)
render_plot(c, "", preedited_path / "merfish_zombie_fov.svg", 3.5, 2.6, x_offset=30,y_offset=10)
render_plot(c, "D", preedited_path / "merfish_zombie_integration_confusion_matrix.svg", 6.4, 2.6, x_offset=25,y_offset=15)
render_plot(c, "E", preedited_path / "merfish_zombie_detection_stats_barplot.svg", 6.4, 3.7, x_offset=25)
render_plot(c, "F", preedited_path / "frac_decoded_barplot.svg", 0, 5.4, x_offset=30,y_offset=15)
render_plot(c, "G",None,3.3, 5.4)
render_plot(c, "H", preedited_path / "merfish_invitro_confusion_matrix.svg", 0, 7.8, x_offset=30,y_offset=15)


# Save canvas
c.save()