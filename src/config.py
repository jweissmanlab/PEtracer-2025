from pathlib import Path
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns

threads = 30
log_path = None

# Names
site_names = {"RNF2":"Edit site 1", "HEK3":"Edit site 2", "EMX1":"Edit site 3"}
edit_names = {'EMX1': {'None': 0,
  'GGACA': 1,
  'ACAAT': 2,
  'CCCTA': 3,
  'AGTAC': 4,
  'CCGAT': 5,
  'CCTTT': 6,
  'ATCAA': 7,
  'ATTCG': 8},
 'RNF2': {'None': 0,
  'ACAGT': 1,
  'ACTTA': 2,
  'TTCCT': 3,
  'TATAT': 4,
  'GTTCA': 5,
  'TGCCA': 6,
  'TCCAA': 7,
  'ACTCC': 8},
 'HEK3': {'None': 0,
  'GATAG': 1,
  'AATCG': 2,
  'GCAAG': 3,
  'GCGCC': 4,
  'CTTTG': 5,
  'ATCAA': 6,
  'CTCTC': 7,
  'ATTTA': 8}}

# MERFISH parameters
min_spot_intensity = 1000

# Default colors
colors = ["black", "#1874CD", "#CD2626","#FFE600","#009E73","#8E0496","#E69F00"]

# Sequential colormap
sequential_colors = [(1, 1, 1)] + [plt.cm.GnBu(i / (256 - 1)) for i in range(256)]
sequential_cmap = mcolors.LinearSegmentedColormap.from_list("GnBu", sequential_colors, N=256)

# Discrete colormap
discrete_colors = {
1:["black"],
2:["#CD2626", "#1874CD"],
3:["#CD2626", "#FFE600", "#1874CD"],
4:["#CD2626", "#FFE600", "#009E73", "#1874CD"],
5:["#CD2626", "#FFE600", "#009E73", "#1874CD", "#8E0496"],
6:["#CD2626", "#E69F00", "#FFE600", "#009E73", "#1874CD", "#8E0496"],
7:["#CD2626", "#E69F00", "#FFE600", "#009E73", "#83A4FF", "#1874CD", "#8E0496"],
8:["#CD2626", "#E69F00", "#FFE600", "#009E73", "#83A4FF", "#1874CD", "#8E0496", "#DB65D2"],
9:["#CD2626", "#E69F00", "#FFE600", "#009E73", "#75F6FC", "#83A4FF", "#1874CD", "#8E0496", "#DB65D2"],
10:["#CD2626", "#E69F00", "#FFE600", "#7BE561", "#009E73", "#75F6FC", "#83A4FF", "#1874CD", "#8E0496", "#DB65D2"],
11:["#FF7D7D", "#CD2626", "#E69F00", "#FFE600", "#7BE561", "#009E73", "#75F6FC", "#83A4FF", "#1874CD", "#8E0496", "#DB65D2"],
12:["#FF7D7D", "#CD2626", "#E69F00", "#FFE600", "#7BE561", "#009E73", "#75F6FC", "#83A4FF", "#1874CD", "#7C0EDD", "#8E0496", "#DB65D2"],
13:["#FF7D7D", "#CD2626", "#E69F00", "#FFE600", "#7BE561", "#009E73", "#75F6FC", "#262C6B", "#83A4FF", "#1874CD", "#7C0EDD", "#8E0496", "#DB65D2"],
14:["#FF7D7D", "#CD2626", "#D34818", "#E69F00", "#FFE600", "#7BE561", "#009E73", "#75F6FC", "#262C6B", "#83A4FF", "#1874CD", "#7C0EDD", "#8E0496", "#DB65D2"],
15:["#FF7D7D", "#CD2626", "#D34818", "#E69F00", "#FFE600", "#7BE561", "#009E73", "#20C4AC", "#75F6FC", "#262C6B", "#83A4FF", "#1874CD", "#7C0EDD", "#8E0496", "#DB65D2"],
16:["#FF7D7D", "#CD2626", "#D34818", "#E69F00", "#FFE600", "#7BE561", "#009E73", "#20C4AC", "#75F6FC", "#262C6B", "#83A4FF", "#1874CD", "#A983F2", "#7C0EDD", "#8E0496", "#DB65D2"],
17:["#FF7D7D", "#CD2626", "#D34818", "#E69F00", "#FFE600", "#7BE561", "#009E73", "#20C4AC", "#75F6FC", "#262C6B", "#83A4FF", "#1874CD", "#A983F2", "#7C0EDD", "#8E0496", "#DB65D2", "#FAC0FF"],
18:["#FF7D7D", "#CD2626", "#7F0303", "#D34818", "#E69F00", "#FFE600", "#7BE561", "#009E73", "#20C4AC", "#75F6FC", "#262C6B", "#83A4FF", "#1874CD", "#A983F2", "#7C0EDD", "#8E0496", "#DB65D2", "#FAC0FF"],
19:["#FF7D7D", "#CD2626", "#7F0303", "#D34818", "#E69F00", "#845C44", "#FFE600", "#7BE561", "#009E73", "#20C4AC", "#75F6FC", "#262C6B", "#83A4FF", "#1874CD", "#A983F2", "#7C0EDD", "#8E0496", "#DB65D2", "#FAC0FF"],
20:["#FF7D7D", "#CD2626", "#7F0303", "#D34818", "#E69F00", "#845C44", "#FFE600", "#7BE561", "#009E73", "#20C4AC", "#75F6FC", "#262C6B", "#83A4FF", "#1874CD", "#A983F2", "#7C0EDD", "#8E0496", "#DB65D2", "#FAC0FF", "#343434"]}
discrete_cmap = {k:sns.color_palette(v) for k,v in discrete_colors.items()}
