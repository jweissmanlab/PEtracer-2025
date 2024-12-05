import sys
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics

# Configure paths
figure_path = Path(__file__).parent
base_path = Path(__file__).parent.parent
barcoding_path = base_path / "barcoding" / "plots"
sys.path.append(str(base_path))

# Load source
from petracer.utils import render_plot

# Make canvas
c = canvas.Canvas(str(figure_path / "s6.pdf"), pagesize=(8.5*72, 11*72))
pdfmetrics.registerFont(TTFont('Arial-Bold', 'Arial_Bold.ttf'))
c.setFont('Arial-Bold', 14)

# Render panels
render_plot(c, "A", barcoding_path / "clone_1_combined_clades.png", 0, 0,y_offset=20)
render_plot(c, "", barcoding_path / "clone_2_combined_clades.png", 5.2, 0,y_offset=20)
render_plot(c, "", barcoding_path / "clone_5_combined_clades.png", 5.8, 5.5,y_offset=20)
render_plot(c, "", barcoding_path / "clone_6_combined_clades.png", 4.8, 3.2,y_offset=20)
render_plot(c, "", barcoding_path / "clone_3_combined_clades.png", 2.5, 5,y_offset=20)
render_plot(c, "B", barcoding_path / "nj_vs_upgma_fmi_scatterplot.svg", 0, 8,y_offset = 30)
render_plot(c, "C", barcoding_path / "puro_distance_comparison_kdeplot.svg", 2, 8)
render_plot(c, "", barcoding_path / "blast_distance_comparison_kdeplot.svg", 4, 8)
render_plot(c, "D", barcoding_path / "ks_comparison_scatterplot.svg", 6.2, 8,y_offset = 30)



# Save canvas
c.save()