import sys
from pathlib import Path
from reportlab.pdfgen import canvas

# Configure paths
figure_path = Path(__file__).parent
base_path = Path(__file__).parent.parent
sim_path = base_path / "lineage_tracer_simulation" / "plots"
sys.path.append(str(base_path))

# Load source
from src.utils import render_panel

# Make canvas
c = canvas.Canvas(str(figure_path / "s1.pdf"), pagesize=(8.5*72, 11*72))
c.setFont("Helvetica", 16)

# Render panels
render_panel(c, "A", None, 0, 0)
render_panel(c, "B", sim_path / "rf_state_distribution_heatmap.svg", 0, 2.5, .75)
render_panel(c, "C", sim_path / "rf_vs_state_distribution.svg", 5.2, 2.5, .75)
render_panel(c, "D", sim_path / "triplets_state_distribution_heatmap.svg", 0, 4.5, .75)
render_panel(c, "E", sim_path / "triplets_vs_state_distribution.svg", 5.2, 4.5, .75)
render_panel(c, "F", sim_path / "rf_vs_parameter.svg", 0, 6.5, .8)
render_panel(c, "G", sim_path / "triplets_vs_parameter.svg", 0, 8.5, .8)

# Save canvas
c.save()