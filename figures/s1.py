import sys
from pathlib import Path
from reportlab.pdfgen import canvas

# Configure paths
figure_path = Path(__file__).parent
base_path = Path(__file__).parent.parent
sim_path = base_path / "lineage_tracer_simulation" / "plots"
sys.path.append(str(base_path))

# Load source
from src.utils import render_plot

# Make canvas
c = canvas.Canvas(str(figure_path / "s1.pdf"), pagesize=(8.5*72, 11*72))
c.setFont("Helvetica", 16)

# Render panels
render_plot(c, "A", sim_path / "rf_state_distribution_heatmap.svg", 0, 0, .85)
render_plot(c, "B", sim_path / "rf_vs_state_distribution.svg", 5.6, 0, .9,y_offset=30)
render_plot(c, "C", sim_path / "triplets_state_distribution_heatmap.svg", 0, 2.5, .85)
render_plot(c, "D", sim_path / "triplets_vs_state_distribution.svg", 5.6, 2.5, .9,y_offset=30)
render_plot(c, "E", sim_path / "rf_vs_parameter.svg", 0, 4.9, .95,y_offset=35)
render_plot(c, "F", sim_path / "triplets_vs_parameter.svg", 0, 7.5, .95)

# Save canvas
c.save()