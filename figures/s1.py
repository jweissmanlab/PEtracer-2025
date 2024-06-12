import sys
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics

# Configure paths
figure_path = Path(__file__).parent
base_path = Path(__file__).parent.parent
sim_path = base_path / "simulation" / "plots"
legend_path = base_path / "legends" / "plots"
sys.path.append(str(base_path))

# Load source
from src.utils import render_plot

# Make canvas
c = canvas.Canvas(str(figure_path / "s1.pdf"), pagesize=(8.5*72, 11*72))
pdfmetrics.registerFont(TTFont('Arial', 'Arial.ttf'))
c.setFont('Arial', 16)

# Render panels
render_plot(c, "A", None, 0, 0)
render_plot(c, "B", sim_path / "rf_heatmap_entropy_vs_states.svg", 3, 0)
render_plot(c, "C", sim_path / "rf_heatmap_edit_frac_vs_states.svg", 5.3, 0)
render_plot(c, "", legend_path / "rf_cbar.svg", 7.7, 0,x_offset=0,y_offset=30)
render_plot(c, "D", sim_path / "rf_parameter_sweep_lineplot.svg", 0, 2.4)
render_plot(c, "D", sim_path / "rf_parameter_sweep_lineplot.svg", 0, 2.4)
render_plot(c, "E", sim_path / "min_characters_lineplot.svg", 0, 4.7)

# Save canvas
c.save()