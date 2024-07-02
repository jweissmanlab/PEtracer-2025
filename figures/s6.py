import sys
from pathlib import Path
from reportlab.pdfgen import canvas

# Configure paths
figure_path = Path(__file__).parent
base_path = Path(__file__).parent.parent
barcoding_path = base_path / "barcoding" / "plots"
sys.path.append(str(base_path))

# Load source
from src.utils import render_plot

# Make canvas
c = canvas.Canvas(str(figure_path / "s6.pdf"), pagesize=(8.5*72, 11*72))
c.setFont("Helvetica", 16)

# Render panels
render_plot(c, "A", barcoding_path / "clone_1_combined_clades.png", 0, 0,y_offset=5)
render_plot(c, "", barcoding_path / "clone_2_combined_clades.png", 5.2, 0,y_offset=5)
render_plot(c, "", barcoding_path / "clone_5_combined_clades.png", 5.6, 5.7,y_offset=5)
render_plot(c, "", barcoding_path / "clone_6_combined_clades.png", 4.8, 3.2,y_offset=5)
render_plot(c, "", barcoding_path / "clone_3_combined_clades.png", 2.3, 5.2,y_offset=5)
render_plot(c, "B", barcoding_path / "nj_vs_upgma_fmi_scatterplot.svg", 0, 8.4)

# Save canvas
c.save()