import torch
from flexfold.core import struct_to_pdb
from openfold.utils.tensor_utils import tensor_tree_map
import numpy as np 
from openfold.np import residue_constants, protein
from openfold.config import model_config
from openfold.model.model import AlphaFold









struct = torch.load("/home/vuillemr/cryofold/cryobench_IgD/pred2/embeddings.pt")
struct = tensor_tree_map(lambda x: x.detach().cpu().numpy(), struct)
struct_to_pdb(struct,"/home/vuillemr/cryofold/cryobench_IgD/test.pdb" )


config = model_config(
    "model_3_multimer_v3", 
    train=True, 
    low_prec=False,
) 
model = AlphaFold(config)



structure_module = StructureModule(
            is_multimer=is_multimer,
            **self.config["structure_module"],
        )
structure_input = {
    "pair": struct["pair"],
    "single": struct["single"]
}
        # Predict 3D structure
outputs = {}
outputs["sm"] = self.structure_module(
    structure_input,
    embedding_expand["aatype"],
    mask=embedding_expand["seq_mask"],
    inplace_safe=inplace_safe,
    _offload_inference=self.globals.offload_inference,
)
outputs["final_atom_positions"] = atom14_to_atom37(
    outputs["sm"]["positions"][-1], embedding_expand
)
outputs["final_atom_mask"] = embedding_expand["atom37_atom_exists"]
outputs["final_affine_tensor"] = outputs["sm"]["frames"][-1]










from cryodrgn import utils, config
z = utils.load_pkl("data/cryofold/particlesSNR1.0/run_fourier_inf_baseline/z.6.pkl")
zdim = z.shape[1]


from cryodrgn import utils, config
from flexfold.models import HetOnlyVAE
from flexfold import dataset




dataset.ImageDataset(
            mrcfile="data/cryofold/cryobench_IgD/IgG-1D/images/snr0.01/099_particles_128.mrcs",
            lazy="False",
            norm=None,
            invert_data=False,
            ind=None,
            keepreal=False,
            window=True,
            datadir=None,
            window_r=0.85,
            max_threads=0,
        )


cfg = config.load("data/cryofold/cryobench_IgD/IgG-1D/images/snr0.01/run/config.yaml")
D = cfg["lattice_args"]["D"]  # image size + 1
zdim = cfg["model_args"]["zdim"]
norm = [float(x) for x in cfg["dataset_args"]["norm"]]
model, lattice = HetOnlyVAE.load(cfg, "data/cryofold/cryobench_IgD/IgG-1D/images/snr0.01/run/weights.0.pkl", device="cuda:0")
model.eval()







import torch
from openfold.np.residue_constants import restypes
from flexfold.core import output_single_pdb
from Bio.PDB.MMCIFParser import MMCIFParser
embeddings=  torch.load("data/cryofold/spike-md/embeddings.pt", map_location="cuda:0")

parser = MMCIFParser(QUIET=True)
structure = parser.get_structure("pdb", "data/cryofold/spike-md/6VSB.cif")
model = next(structure.get_models())  # usually only 1 model
# Extract all residues (modeled)
modeled = set()
for chain in model:
    if chain.id == "A":
        for res in chain:
            if res.id[0] == " ":  # ignore hetero/water
                modeled.add(res.id[1])  # resseq number

modeled = torch.tensor(list(modeled), device="cuda:0")


resisd = embeddings["residue_index"]

crop = (resisd<=(1146-1)) * (resisd>=(27-1))
mask = torch.tensor([1 if i in modeled else 0 for i in embeddings["residue_index"]], device="cuda:0")

output_single_pdb(
    embeddings["final_atom_positions"][crop].cpu().numpy(),
    embeddings["aatype"][crop].cpu().numpy(),
    (mask[..., None]*  embeddings["final_atom_mask"])[crop].cpu().numpy(),
    "data/cryofold/spike-md/test.pdb",
    embeddings["asym_id"][crop].cpu().numpy(),
    embeddings["residue_index"][crop].cpu().numpy(),

)

output_single_pdb(
    embeddings["final_atom_positions"][crop].cpu().numpy(),
    embeddings["aatype"][crop].cpu().numpy(),
    (embeddings["final_atom_mask"])[crop].cpu().numpy(),
    "data/cryofold/spike-md/test_unmask.pdb",
    embeddings["asym_id"][crop].cpu().numpy(),
    embeddings["residue_index"][crop].cpu().numpy(),

)

embeddings_new = {
    "aatype": embeddings["aatype"][crop],
    "seq_mask": mask[crop],
    "pair": embeddings["pair"][crop, :, :][:, crop, :],
    "single":  embeddings["single"][crop, :],
    "final_atom_mask": (mask[..., None]* embeddings["final_atom_mask"])[crop, :],
    "final_atom_positions":embeddings["final_atom_positions"][crop, :,:],
    "residx_atom37_to_atom14":embeddings["residx_atom37_to_atom14"][crop, :],
    "residx_atom14_to_atom37":embeddings["residx_atom14_to_atom37"][crop, :],
    "atom37_atom_exists":embeddings["atom37_atom_exists"][crop, :],
    "atom14_atom_exists":embeddings["atom14_atom_exists"][crop, :],
    "residue_index": embeddings["residue_index"][crop],
    "asym_id":  embeddings["asym_id"][crop],
}

output_single_pdb(
    embeddings_new["final_atom_positions"].cpu().numpy(),
    embeddings_new["aatype"].cpu().numpy(),
    (embeddings_new["final_atom_mask"]).cpu().numpy(),
    "data/cryofold/spike-md/test.pdb",
    embeddings_new["asym_id"].cpu().numpy(),
    embeddings_new["residue_index"].cpu().numpy(),

)

torch.save({k: v.cpu() for k, v in embeddings_new.items()}, "data/cryofold/spike-md/embeddings_crop_mask.pt")


import torch
from openfold.np.residue_constants import restypes
from flexfold.core import output_single_pdb
embeddings=  torch.load("data/cryofold/jillsData/pred2/embeddings.pt")
embeddings_new = {
    "aatype": embeddings["aatype"],
    "seq_mask": embeddings["seq_mask"],
    "pair": embeddings["pair"],
    "single":  embeddings["single"],
    "final_atom_mask": embeddings["final_atom_mask"],
    "final_atom_positions":embeddings["final_atom_positions"],
    "residx_atom37_to_atom14":embeddings["residx_atom37_to_atom14"],
    "residx_atom14_to_atom37":embeddings["residx_atom14_to_atom37"],
    "atom37_atom_exists":embeddings["atom37_atom_exists"],
    "atom14_atom_exists":embeddings["atom14_atom_exists"],
    "residue_index": embeddings["residue_index"],
    "asym_id":  embeddings["asym_id"],
}
torch.save({k: v.cpu() for k, v in embeddings_new.items()}, "data/cryofold/jillsData/pred2/embeddings_fixed.pt")


import pandas as pd
import io

def read_star_multi(star_path):
    blocks = {}
    with open(star_path, "r") as f:
        lines = f.readlines()

    current_block = None
    headers = []
    data_lines = []
    for line in lines:
        line = line.strip()
        if line.startswith("data_"):
            # flush previous block if any
            if current_block and headers:
                df = pd.read_csv(
                    io.StringIO("\n".join(data_lines)),
                    delim_whitespace=True,
                    names=headers,
                )
                blocks[current_block] = df.to_dict(orient="list")
            # reset for new block
            current_block = line
            headers, data_lines = [], []
        elif line.startswith("_rln"):
            headers.append(line.split()[0])
        elif headers and line and not line.startswith("#"):
            data_lines.append(line)

    # flush last block
    if current_block and headers:
        df = pd.read_csv(
            io.StringIO("\n".join(data_lines)),
            delim_whitespace=True,
            names=headers,
        )
        blocks[current_block] = df.to_dict(orient="list")


    print("========================== Summary ==========================")
    for k,v in blocks.items():
        print(k)
        print("|")
        for k2,v2 in v.items():
            print("|--> %s, len=%i, dtype=%s"%(k2, len(v2), type(next(iter(v2)))))
        print("")

    return blocks

def write_star_from_dict(star_dict, out_path, data_block="data_particles"):
    """
    Write a RELION .star file from a dict of lists.

    Parameters
    ----------
    star_dict : dict
        Keys are RELION column names (e.g. '_rlnImageName'),
        values are lists of column entries (all same length).
    out_path : str
        Where to write the .star file.
    data_block : str
        Name of the data block (default: "data_particles").
    """
    keys = list(star_dict.keys())
    n_rows = len(next(iter(star_dict.values())))  # length of first column

    with open(out_path, "w") as f:
        # block header
        f.write(f"{data_block}\n\n")
        f.write("loop_\n")
        # headers with column numbers
        for i, k in enumerate(keys, start=1):
            f.write(f"{k} #{i}\n")

        # write rows
        for row_idx in range(n_rows):
            row = [str(star_dict[k][row_idx]) for k in keys]
            f.write(" ".join(row) + "\n")


star = read_star_multi("/home/vuillemr/flexfold/data/cryofold/jillsData/particles/particles_ctf.star")


keywords = [
    ["_anglePsi","_rlnAnglePsi"],
    ["_angleRot","_rlnAngleRot"],
    ["_angleTilt","_rlnAngleTilt"],
    ["_image","_rlnImageName"],
    ["_shiftX","_rlnOriginXAngst"],
    ["_shiftY","_rlnOriginYAngst"],
    ["_ctfVoltage","_rlnVoltage"],
    ["_ctfSphericalAberration","_rlnSphericalAberration"],
    ["_ctfSamplingRate","_rlnDetectorPixelSize"],
    ["_magnification","_rlnMagnification"],
    ["_ctfDefocusU","_rlnDefocusU"],
    ["_ctfDefocusV","_rlnDefocusV"],
    ["_ctfQ0","_rlnAmplitudeContrast"]
]

xmp2rln_kw = {k:v for k,v in keywords}
rln2xmp_kw = {v:k for k,v in keywords}

def rln2xmp(star_dict):
    return {rln2xmp_kw[k]:v for k,v in star_dict.items() if k in rln2xmp_kw}
    
def xmp2rln(star_dict):
    return {xmp2rln_kw[k]:v for k,v in star_dict.items() if k in rln2xmp_kw}
    


write_star_from_dict(rln2xmp(star["data_particles"]), "/home/vuillemr/flexfold/data/cryofold/jillsData/particles/particles.xmp")



########################################################################""
import torch 
from flexfold.models import HetOnlyVAE
from cryodrgn import config
from cryodrgn.utils import load_pkl
import matplotlib.pyplot as plt

config_file = "/home/vuillemr/cryofold/cryobench_IgD/IgG-1D/images/snr0.01/run2/config.yaml"
weight_file = "/home/vuillemr/cryofold/cryobench_IgD/IgG-1D/images/snr0.01/run2/weights.20.pkl"

##################################################""
# LOAD MODEL and WEIGHTS
##################################################""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = config.load(config_file)
D = cfg["lattice_args"]["D"]  # image size + 1
zdim = cfg["model_args"]["zdim"]
norm = [float(x) for x in cfg["dataset_args"]["norm"]]
model, lattice = HetOnlyVAE.load(cfg,weight_file, device=device)
model.eval()

##################################################""
# LOAD Z
##################################################""
z = load_pkl("/home/vuillemr/cryofold/cryobench_IgD/IgG-1D/images/snr0.01/run2/z.20.pkl")
zdim = z.shape[-1]
##################################################""
# PLOT Z
##################################################""
kwargs = {
    "c":[i for i in range(len(z))],
    "cmap":"jet",
    "alpha":0.5,
    "s":0.1
}
fig, ax = plt.subplots(zdim-1,zdim-1, figsize=(10,10), layout="constrained")
for i in range(zdim-1):
    for j in range(1, zdim):
        if not i>=j:
            ax[j-1,i].scatter(z[:,i], z[:,j], **kwargs)
            ax[j-1,i].set_xlabel("Z%i"%i, fontsize=15)
            ax[j-1,i].set_ylabel("Z%i"%j, fontsize=15)
        else:
            ax[j-1,i].set_visible(False)

fig.savefig("/home/vuillemr/cryofold/cryobench_IgD/IgG-1D/images/snr0.01/run2/analysis/test_z.svg")

##################################################""
# DIMENTIONALITY REDUCTION
##################################################""

from umap import UMAP
from sklearn.decomposition import PCA
import numpy as np
from sklearn.cluster import KMeans

import numpy as np
import itertools
import tqdm
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import numpy as np
from scipy.spatial import distance_matrix

def tsp_nearest_neighbor(points):
    """Simple nearest neighbor heuristic for TSP."""
    n = len(points)
    dist = distance_matrix(points, points)
    unvisited = set(range(n))
    path = [0]  # start at point 0 (arbitrary)
    unvisited.remove(0)
    
    while unvisited:
        last = path[-1]
        next_idx = min(unvisited, key=lambda j: dist[last, j])
        path.append(next_idx)
        unvisited.remove(next_idx)
    
    return path

def tsp_path_length(points, path):
    """Compute length of a given TSP path."""
    return sum(np.linalg.norm(points[path[i]] - points[path[i-1]])
               for i in range(1, len(path)))

# --- Optional: 2-opt optimization ---
def two_opt(points, path):
    """Try to improve TSP path using 2-opt."""
    best = path
    best_len = tsp_path_length(points, best)
    improved = True
    
    while improved:
        improved = False
        for i in range(1, len(path) - 2):
            for j in range(i+1, len(path)):
                if j - i == 1:  # consecutive edges, skip
                    continue
                new_path = best[:i] + best[i:j][::-1] + best[j:]
                new_len = tsp_path_length(points, new_path)
                if new_len < best_len:
                    best = new_path
                    best_len = new_len
                    improved = True
        path = best
    return best

def solve_tsp(points):
    n = len(points)
    dist_matrix = np.linalg.norm(points[:,None] - points[None,:], axis=2)
    
    # Setup routing
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)
    
    def distance_callback(i, j):
        return int(dist_matrix[manager.IndexToNode(i)][manager.IndexToNode(j)] * 1000)
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # Search
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_params.time_limit.seconds = 10
    
    solution = routing.SolveWithParameters(search_params)
    
    # Extract route
    route = []
    idx = routing.Start(0)
    while not routing.IsEnd(idx):
        route.append(manager.IndexToNode(idx))
        idx = solution.Value(routing.NextVar(idx))
    return route


def order_clusters_greedy(centers):
    remaining = list(range(len(centers)))
    order = [remaining.pop(0)]  # start from first center
    while remaining:
        last = centers[order[-1]]
        # find nearest remaining center
        next_idx = min(remaining, key=lambda i: np.linalg.norm(last - centers[i]))
        order.append(next_idx)
        remaining.remove(next_idx)
    return order

def brute_force_tsp(points):
    n = len(points)
    best_order = None
    best_length = float("inf")
    
    for perm in tqdm.tqdm(itertools.permutations(range(n))):
        length = np.sum(np.linalg.norm(points[perm[i]] - points[perm[i-1]]) 
                        for i in range(1, n))
        if length < best_length:
            best_length = length
            best_order = perm
    return np.array(best_order)

def tsp_bidirectional(points):
    """Build a TSP tour by growing in both directions from each possible start."""
    n = len(points)
    dist = distance_matrix(points, points)
    best_path, best_len = None, float("inf")
    
    for start in range(n):
        left = [start]
        right = []
        unvisited = set(range(n)) - {start}
        
        # Alternate: choose nearest to either end
        while unvisited:
            # candidates from leftmost and rightmost ends
            left_end, right_end = left[0], (right[-1] if right else left[-1])
            
            cand_left = min(unvisited, key=lambda j: dist[left_end, j])
            cand_right = min(unvisited, key=lambda j: dist[right_end, j])
            
            if dist[left_end, cand_left] < dist[right_end, cand_right]:
                left.insert(0, cand_left)
                unvisited.remove(cand_left)
            else:
                right.append(cand_right)
                unvisited.remove(cand_right)
        
        path = left + right
        length = sum(np.linalg.norm(points[path[i]] - points[path[i-1]]) for i in range(1, n))
        
        if length < best_len:
            best_path, best_len = path, length
    
    return np.array(best_path)

def mst_traversal(centers):
    dist = squareform(pdist(centers))
    mst = minimum_spanning_tree(dist)
    mst_dense = mst.toarray()              # convert to dense numpy array
    mst_dense = mst_dense + mst_dense.T  # make symmetric
    k = mst_dense.shape[0]
    visited = [False] * k
    order = []

    # find a leaf node to start (node with only 1 connection)
    degrees = (mst_dense > 0).sum(axis=1)
    start = np.argmin(degrees)

    def dfs(node):
        visited[node] = True
        order.append(node)
        neighbors = np.where(mst_dense[node] > 0)[0]
        for n in neighbors:
            if not visited[n]:
                dfs(n)

    dfs(start)
    return order

# dimred = UMAP(n_components=2, n_neighbors=50, min_dist=0.1, random_state=42)
dimred = PCA(n_components=2)
data = dimred.fit_transform(z)

n_clusters=9
k_means = KMeans(init='k-means++', n_clusters=n_clusters)
k_means.fit(data)
centers = k_means.cluster_centers_ 
labels = k_means.labels_

# ordered_idx = mst_traversal(centers)
# ordered_idx = brute_force_tsp(centers)
# ordered_idx = order_clusters_greedy(centers)
# ordered_idx = np.array(two_opt(centers, tsp_nearest_neighbor(centers)) )
ordered_idx = tsp_bidirectional(centers)

traj_center = centers[ordered_idx]
old_to_new = {old: new for new, old in enumerate(ordered_idx)}
traj_labels = np.array([old_to_new[l] for l in labels])

cmap = "plasma"
fig, ax = plt.subplots(1,1, figsize=(10,10))
ax.scatter(data[:,0], data[:,1], cmap=cmap, alpha=0.5, c=traj_labels, s =10)
ax.plot(traj_center[:,0], traj_center[:,1], "-", c="black")
ax.scatter(traj_center[:,0], traj_center[:,1],s=200, cmap=cmap, c=[i for i in range(n_clusters)], edgecolors="black")
fig.savefig("/home/vuillemr/cryofold/cryobench_IgD/IgG-1D/images/snr0.01/run2/analysis/test_umap.png")


##################################################""
# output models
##################################################""
from flexfold.core import struct_to_pdb
from openfold.utils.tensor_utils import tensor_tree_map
from cryodrgn.mrcfile import write_mrc

indices = [traj_labels == i for i in range(n_clusters)]
z_traj_avg = [np.mean(z[i], axis=0) for i in indices]
z_traj_avg = torch.tensor(z_traj_avg, device=device)

model.to(device)

for i in range(n_clusters):
    prefix = "/home/vuillemr/cryofold/cryobench_IgD/IgG-1D/images/snr0.01/run2/analysis/test%s"%str(i+1).zfill(5)
    vol, struct = model.decoder.eval_volume(lattice.coords, D, lattice.extent,norm, z_traj_avg[i])
    crd = struct["final_atom_positions"]
    crd = crd @ model.decoder.rot_init + model.decoder.trans_init[..., None, :]
    struct["final_atom_positions"] = crd
    struct_to_pdb(tensor_tree_map(lambda x: x.detach().cpu().numpy()[-1], struct), 
                  prefix+".pdb")

    write_mrc(prefix+".mrc", np.array(vol.cpu()).astype(np.float32), Apix=3.0)

