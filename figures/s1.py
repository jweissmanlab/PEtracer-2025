import sys
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics

# Configure paths
figure_path = Path(__file__).parent
base_path = Path(__file__).parent.parent
sim_path = base_path / "simulation" / "plots"
insert_path = base_path / "insert_selection" / "plots"

legend_path = base_path / "legends" / "plots"
sys.path.append(str(base_path))

# Load source
from src.utils import render_plot

# Make canvas
c = canvas.Canvas(str(figure_path / "s1.pdf"), pagesize=(8.5*72, 11*72))
pdfmetrics.registerFont(TTFont('Arial-Bold', 'Arial_Bold.ttf'))
c.setFont('Arial-Bold', 14)

# Render panels
render_plot(c, "A", sim_path / "rf_heatmap_edit_frac_vs_states.svg", 0, 0)
render_plot(c, "B", sim_path / "rf_parameter_sweep_lineplot.svg", 2.3, 0,x_offset = 22)
render_plot(c, "C", sim_path / "frac_over_time_lineplot.svg", 0, 2.8,x_offset = 20,y_offset=20)
render_plot(c, "D", sim_path / "log_edit_rate_lineplot.svg", 2.9, 2.8,x_offset = 20,y_offset=20)
render_plot(c, "E", insert_path / "correct_frac_vs_length.svg", 6, 2.8,x_offset = 25,y_offset=20)

# Save canvas
c.save()