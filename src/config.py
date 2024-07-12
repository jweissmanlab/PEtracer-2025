from pathlib import Path
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns

threads = 30
log_path = None

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
min_edit_prob = 0.5
default_decoder = "v4_edit_decoder.pkl"
img_paths = {
    "preedited_merfish_invitro": {
        "path": "/lab/weissman_imaging/puzheng/PE_LT/20240426-fullyEdited4T1_ingel_IntBCv2_combinedEditv3/",
        "analysis": "/lab/weissman_imaging/wcolgan/MERFISH_Data/20240426-fullyEdited4T1_ingel_IntBCv2_combinedEditv3/Analysis/",
        "file_pattern": "{series}/Conv_zscan_{fov:02d}.dax"},
    "preedited_merfish_invivo": {
        "path": "/lab/weissman_imaging/puzheng/PE_LT/20240424-F242dpec_T7afterMerfish/",
        "analysis": "/lab/weissman_imaging/wcolgan/20240424-F242dpec_T7afterMerfish/Analysis/",
        "file_pattern": "{series}/Conv_zscan_{fov:03d}.dax"},
    "preedited_merfish_zombie": {
        "path": "/lab/weissman_imaging/puzheng/PE_LT/20240508-4T1fullyEdited_zombie_IntBCv2new_editv3/",
        "analysis": "/lab/weissman_imaging/wcolgan/20240508-4T1fullyEdited_zombie_IntBCv2new_editv3/Analysis/",
        "file_pattern": "{series}/Conv_zscan_{fov:02d}.dax"},
}
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