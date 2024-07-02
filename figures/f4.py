import sys
from pathlib import Path
from reportlab.pdfgen import canvas

# Configure paths
figure_path = Path(__file__).parent
base_path = Path(__file__).parent.parent
preedited_path = base_path / "preedited_validation" / "plots"
sys.path.append(str(base_path))

# Load source
from src.utils import render_plot

# Make canvas
c = canvas.Canvas(str(figure_path / "f4.pdf"), pagesize=(8.5*72, 11*72))
c.setFont("Helvetica", 16)

# Render panels
render_plot(c, "A", None, 0, .9)
render_plot(c, "B", preedited_path / "sc_allele_heatmap.png", 2.5, .9)
render_plot(c, "C", preedited_path / "clone_edit_whitelist.svg", 6.4, .9)
render_plot(c, "D", preedited_path / "sc_detection_violin.svg", 6.4, 2.15,y_offset=35)
render_plot(c, "E", preedited_path / "invitro_cell_box.svg", 0, 4.2)
render_plot(c, "", preedited_path / "invitro_cell_z_projection.svg", 0, 6)
render_plot(c, "", preedited_path / "invitro_cell_y_projection.svg", 1.5, 6)
render_plot(c, "F", preedited_path / "invitro_cell_integration_spots.svg", 2.5, 4.2)
render_plot(c, "G", preedited_path / "invitro_cell_edit_spots.svg", 2.5, 6.3)
render_plot(c, "H", preedited_path / "invitro_merfish_detection_violin.svg", 0, 8.3,y_offset=32)
render_plot(c, "I", preedited_path / "invitro_merfish_confusion.png", 1.8, 8.3)

# Save canvas
c.save()