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
c = canvas.Canvas(str(figure_path / "s7.pdf"), pagesize=(8.5*72, 11*72))
c.setFont("Helvetica", 16)

# Render panels
render_plot(c, "A", preedited_path / "invitro_fov_box.svg", 0, 0)
render_plot(c, "", preedited_path / "invitro_fov_labeled_spots.png", 1.2, .93)
render_plot(c, "B", preedited_path / "invitro_precision_recall_threshold.svg", 0, 8.3)
render_plot(c, "C", preedited_path / "invitro_merfish_n_cells.svg", 2.5, 8.3)

# Save canvas
c.save()