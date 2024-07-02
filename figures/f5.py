import sys
from pathlib import Path
from reportlab.pdfgen import canvas

# Configure paths
figure_path = Path(__file__).parent
base_path = Path(__file__).parent.parent
barcoding_path = base_path / "barcoding" / "plots"
sys.path.append(str(base_path))

# Load source
from src.utils import render_plot, convert_svg_text

# Make canvas
c = canvas.Canvas(str(figure_path / "f5.pdf"), pagesize=(8.5*72, 11*72))
c.setFont("Helvetica", 16)

# Render panels
#convert_svg_text(barcoding_path / "barcoding_schematic.svg")
render_plot(c, "A", barcoding_path / "barcoding_schematic.svg",0, 0,y_offset=5,x_offset=30)
render_plot(c, "B", barcoding_path / "clone_4_with_characters.png", 0, 2)
render_plot(c, "C", barcoding_path / "clone_4_combined_clades.png", 3.5, 0)
#D allele vs phylogenetic distance
render_plot(c, "E", barcoding_path / "clone_fmi_violin.svg", 0, 4.3)
render_plot(c, "F", barcoding_path / "clone_fmi_vs_characters_lineplot.svg", 2.1, 4.3)
render_plot(c, "G", barcoding_path / "clone_fmi_vs_detection_lineplot.svg", 4.2, 4.3)




# Save canvas
c.save()