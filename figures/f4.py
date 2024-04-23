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
render_plot(c, "F", preedited_path / "example_cell_z_projection.svg", 0, 4.5)
render_plot(c, None, preedited_path / "example_cell_x_projection.svg", 0, 6.8)
render_plot(c, "G", preedited_path / "example_integration_spots.svg", 2.5, 4.5)
render_plot(c, "H", preedited_path / "example_edit_spots.svg", 2.5, 6.5)
render_plot(c, "I", preedited_path / "imaging_performance.svg", 0, 8.3,y_offset=35)
render_plot(c, "J", preedited_path / "imaging_edit_confusion.svg", 2, 8.3)

# Save canvas
c.save()