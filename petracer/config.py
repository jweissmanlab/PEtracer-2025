from pathlib import Path
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns

threads = 30
log_path = None

# Paths
base_path = Path(__file__).resolve().parent.parent

# Names
preedited_clone_names = {0:"Normal",1:"Clone 1",2:"Clone 2",3:"Clone 3",4:"Clone 4"}
site_ids = {"RNF2":1,"HEK3":2,"EMX1":3}
site_names = {"RNF2":"Edit site 1", "HEK3":"Edit site 2", "EMX1":"Edit site 3"}
edit_ids = {'EMX1': {'None': 0,
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
edit_names = {}
for site in edit_ids:
    edit_names[site] = {edit:f"LM {edit_ids[site][edit]}" for edit in edit_ids[site] if edit != "None"}
    edit_names[site].update({"None":"Unedited"})

# MERFISH parameters
min_edit_prob = 0.7
default_decoder = "v4_edit_decoder.pkl"
fov_size = 246.528

# Default colors
colors = ['black','#1874CD','#CD2626','#FFE600','#009E73','#8E0496','#E69F00','#83A4FF','#DB65D2','#75F6FC','#7BE561','#FF7D7D','#7C0EDD','#262C6B','#D34818','#20C4AC','#A983F2','#FAC0FF','#7F0303','#845C44','#343434']

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

# Edit colors
edit_palette = {"-1":"white"}
edit_palette.update({str(i):discrete_cmap[8][i-1] for i in range(1,9)})
edit_palette.update({"0":"lightgray","9":"#505050"})
edit_cmap = mcolors.ListedColormap(["white","lightgray"] + list(discrete_cmap[8]))
full_edit_cmap = mcolors.ListedColormap(["white","lightgray"] + list(discrete_cmap[8]) + ["#505050"])

# Preedited clone colors
preedited_clone_colors = {"0":"lightgray","1":"#CD2626","2":"#FFE600","3":"#009E73","4":"#1874CD"}

subtype_palette = {
    'Malignant': colors[1],
    'ARG1 macrophage': colors[2],
    'CD11c macrophage': colors[6],
    'ALOX15 macrophage': colors[15],
    'Alveolar macrophage': colors[20],
    'Exhausted CD8 T cell' : colors[9],
    'CD4 T cell' : colors[18],
    'Treg' : colors[12],
    'B cell': colors[8],
    'cDC': colors[16],
    'Neutrophil': colors[7],
    'NK': colors[11],
    'pDC':colors[17],
    'Endothelial': colors[10],
    'CAP1 endothelial': colors[10],
    'CAP2 endothelial': colors[18],
    'Tumor endothelial': colors[15],
    'Cancer fibroblast': colors[3],
    'Alveolar fibroblast 1': colors[4],
    'Alveolar fibroblast 2': colors[19],
    'AT1/AT2': colors[5],
    'Club cell': colors[14],
}

# subtype abbreviations
subtype_abbr = {'Malignant': 'Malig.',
    'Macrophage': 'Mac.',
    'ARG1 macrophage': 'ARG1 Mac.',
    'CD11c macrophage': 'CD11c Mac.',
    'ALOX15 macrophage': 'ALOX15 Mac.',
    'Alveolar macrophage':'Alveolar Mac.',
    'Exhausted CD8 T cell': 'CD8+ T',
    'CD4 T cell': 'CD4+ T',
    'Treg': 'Treg',
    'B cell': 'B cell',
    'cDC': 'cDC',
    'Neutrophil': 'Neu.',
    'NK': 'NK',
    'pDC': 'pDC',
    'Endothelial': 'Endo.',
    'CAP1 endothelial': 'CAP1 Endo.',
    'CAP2 endothelial': 'CAP2 Endo.',
    'Tumor endothelial': 'Tumor Endo.',
    'Fibroblast': 'Fibro.',
    'Cancer fibroblast': 'CAF',
    'Alveolar fibroblast 1': 'AF1',
    'Alveolar fibroblast 2': 'AF2',
    'AT1/AT2': 'AT1/AT2',
    'Club cell': 'Club cell'}

# Phase palette
phase_palette = {
    'G0/G1': colors[13],
    'G2/M': colors[10],
    'S':  colors[5]
}

# Module palette
module_palette = {
    "1":'#009E73',
    "2":'#FFE600',
    "3":"#1874CD",
    "4":'#CD2626'}

# Leiden palette
leiden_palette = {
    "1":colors[7],
    "2":colors[6],
    "3":colors[10],
    "4":colors[8]
}

def get_clade_palette(tdata,key = "clade"):
    """Get a clade palette for a given tdata object."""
    clades = sorted(tdata.obs[key].dropna().unique(), key=lambda x: int(x))
    n_clades = len(clades)
    return {str(clades[i]):discrete_cmap[n_clades][i] for i in range(n_clades)}


def set_theme(figsize=(3, 3), dpi=200):
    """Set the default style for the plots"""
    plt.style.use(base_path / "plot.mplstyle")
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["figure.dpi"] = dpi
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=colors[1:])


def get_paths(folder):
    """Get the paths for the data, plots, and results folders"""
    folder = Path(folder)
    if "/" in str(folder):
        base_path = folder.parent
        folder = folder.name
    else:
        base_path = Path(__file__).resolve().parent.parent
    return base_path, base_path / folder / "data", base_path / folder / "plots", base_path / folder / "results"