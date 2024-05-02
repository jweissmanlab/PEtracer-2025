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
c = canvas.Canvas(str(figure_path / "s8.pdf"), pagesize=(8.5*72, 11*72))
c.setFont("Helvetica", 16)

# Render panels
render_plot(c, "A", preedited_path / "invivo_fov_box.svg", 0, 0)
render_plot(c, "", preedited_path / "invivo_fov_labeled_spots.svg", 1.2, .9)
render_plot(c, "B", preedited_path / "invivo_merfish_detection_violin.svg", 0, 8.3,y_offset=32)
render_plot(c, "C", preedited_path / "invivo_merfish_confusion.svg", 1.8, 8.3)

# Save canvas
c.save()