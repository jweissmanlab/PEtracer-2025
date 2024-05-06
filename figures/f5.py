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
c = canvas.Canvas(str(figure_path / "f5.pdf"), pagesize=(8.5*72, 11*72))
c.setFont("Helvetica", 16)

# Render panels
render_plot(c, "A", None, 0, 0)
render_plot(c, "B", barcoding_path / "example_tree_with_edits.png", 0, 1.6,scale = 1.1,y_offset=15)
render_plot(c, "C", barcoding_path / "example_tree.svg", 4, 1.6, y_offset=5,x_offset=0)
render_plot(c, "D", barcoding_path / "fmi_violin.svg", 0, 4)
render_plot(c, "E", barcoding_path / "site_edit_rates.svg", 2, 4)



# Save canvas
c.save()