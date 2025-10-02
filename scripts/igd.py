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
# FSC
##################################################""

from cryodrgn.commands_utils.fsc import get_fsc_curve, get_fsc_thresholds,calculate_cryosparc_fscs
from cryodrgn.commands_utils.plot_fsc import create_fsc_plot
from cryodrgn.mrcfile import parse_mrc, write_mrc
import torch
import numpy as np
import matplotlib as plt
from flexfold.core import aatype_to_flat_coefs, struct_to_pdb
from openfold.utils.tensor_utils import tensor_tree_map

import numpy as np
import glob
from Bio.SVDSuperimposer import SVDSuperimposer
from Bio.PDB import PDBParser, Superimposer, is_aa
import argparse
import torch 
from flexfold.models import HetOnlyVAE
from cryodrgn import config
from cryodrgn.utils import load_pkl
import matplotlib.pyplot as plt
import itertools
from Bio.PDB import PDBParser, Superimposer, is_aa
from Bio.PDB.Polypeptide import PPBuilder
from Bio import pairwise2
from Bio.Data.IUPACData import protein_letters_3to1

def seq_from_chain(chain):
    """Return (seq, atoms) for a chain."""
    ppb = PPBuilder()
    seq, atoms = "", []
    for pp in ppb.build_peptides(chain):
        for residue in pp:
            if is_aa(residue, standard=True) and "CA" in residue:
                seq += protein_letters_3to1.get(residue.resname.capitalize(), "X")
                atoms.append(residue["CA"])
    return seq, atoms


def align_atoms(seq1, atoms1, seq2, atoms2):
    """Align two chains by sequence and return matched CA atoms."""
    aln = pairwise2.align.globalxx(seq1, seq2, one_alignment_only=True)[0]
    aln1, aln2 = aln.seqA, aln.seqB

    matched1, matched2 = [], []
    i1 = i2 = 0
    for a1, a2 in zip(aln1, aln2):
        atom1 = atom2 = None
        if a1 != "-":
            atom1 = atoms1[i1]; i1 += 1
        if a2 != "-":
            atom2 = atoms2[i2]; i2 += 1
        if atom1 and atom2:
            matched1.append(atom1)
            matched2.append(atom2)
    return matched1, matched2

def rmsd_best_permutation(struct1, struct2):
    """Compute RMSD between two structures, testing all chain permutations."""
    chains1 = list(struct1[0])  # first model only
    chains2 = list(struct2[0])
    if len(chains1) != len(chains2):
        raise ValueError("Different number of chains — need special handling.")

    # Extract seqs and atoms
    seq_atoms1 = [seq_from_chain(c) for c in chains1]
    seq_atoms2 = [seq_from_chain(c) for c in chains2]

    best_rmsd, best_perm, best_rotran = float("inf"), None, None

    # Try all chain permutations
    for perm in itertools.permutations(range(len(chains2))):
        all1, all2 = [], []
        for i, j in enumerate(perm):
            s1, a1 = seq_atoms1[i]
            s2, a2 = seq_atoms2[j]
            m1, m2 = align_atoms(s1, a1, s2, a2)
            all1.extend(m1)
            all2.extend(m2)

        if not all1:
            continue

        sup = Superimposer()
        sup.set_atoms(all1, all2)
        if sup.rms < best_rmsd:
            best_rmsd = sup.rms
            best_perm = perm
            best_rotran = sup.rotran

    return best_rmsd, best_perm, best_rotran

# DEFs
apix = 3.0
res = 3.0
sigma = (res/(apix))
z = load_pkl("/home/vuillemr/cryofold/cryobench_IgD/IgG-1D/images/snr0.01/run2/z.20.pkl")
nz = z.shape[0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from scripts.train import LitDataModule, LitHetOnlyVAE, add_args

parser = argparse.ArgumentParser()
parser= add_args(parser)

args = parser.parse_args([
"../cryofold/cryobench_IgD/IgG-1D/images/snr0.01/sorted_particles.128.txt"
, "--poses"
, "../cryofold/cryobench_IgD/IgG-1D/images/snr0.01/particles.pkl"
, "--ctf"
, "../cryofold/cryobench_IgD/IgG-1D/images/snr0.01/ctf.pkl"
, "-n"
, '100'
, "-o"
, "../cryofold/cryobench_IgD/IgG-1D/images/snr0.01/run2"
, "--pixel_size"
, '3.0'
, "--sigma"
, '1.0'
, "--quality_ratio"
, '5.0'
, "--embedding_path"
, "/home/vuillemr/cryofold/cryobench_IgD/pred2/embeddings.pt"
, "--initial_pose_path"
, "../cryofold/cryobench_IgD/IgG-1D/images/snr0.01/initial_pose_auto.pt"
, "--af_checkpoint_path"
, "../openfold/openfold/resources/params/params_model_3_multimer_v3.npz"
, "--batch-size"
, '1'
, "--num-workers"
, '0'
, "--zdim"
, '4'
, "--enc-dim"
, '256'
, "--enc-layers"
, '3'
, "--dec-dim"
, '256'
, "--dec-layers"
, '3'
, "--domain"
, "hartley"
, "--all_atom"
, "--multimer"
, "--load"
, "../cryofold/cryobench_IgD/IgG-1D/images/snr0.01/run2/weights.20.pkl"
])


datamodule = LitDataModule(args)
datamodule.setup()
model = LitHetOnlyVAE(args, datamodule.imageDataset.D, datamodule.imageDataset.N)
if args.load:
    checkpoint = torch.load(args.load)
    exclude_prefixes = [
        "lattice",
        "decoder.embeddings",
    ]
    exclude_exact = {"decoder.rot_init", "decoder.trans_init"}
    filtered_state_dict = {
        k: v for k, v in checkpoint["model_state_dict"].items()
        if not any(k.startswith(p) for p in exclude_prefixes)
        and k not in exclude_exact
    }
    # now load
    missing, unexpected = model.model.load_state_dict(filtered_state_dict, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)
model.to(device)


# z_mu_all = []
# for batch in enumerate(datamodule.imageDataset):

#     ind = 3500
#     ii = int(ind*8)
#     iii = int(np.floor(ind*8 / 1000))
#     batch = datamodule.imageDataset[ii]

#     print(batch[-1])

#     batch =  [b.to(device) if isinstance(b, torch.Tensor) else torch.tensor(b).to(device)[None] for b in batch]
#     # batch= next(iter(datamodule.train_dataloader()))
#     # batch =  [b.to(device) for b in batch]
#     iii = int(torch.floor(batch[-1]/1000).item())

#     y, y_real, rot, tran, c = model.prepare_batch(batch)
#     z_mu, z_logvar = model.run_encoder(y,c)

#     z_mu_all.append(z_mu.detach().cpu().numpy())

#     y_recon,y_recon_real, mask, struct = model.run_decoder(z_mu,rot, c)
#     model.write_debug(struct, mask, y, y_real, y_recon, y_recon_real, 100)

import pickle
outputs = {
    "fsc": [],
    "fscauc": [],
    "fscres": [],
    "rmsd": []
}
for ind in range(len(z)):
    iii = int(np.floor(ind*8 / 1000))
    print(ind, "/", len(z))
    v,s = model.model.decoder.eval_volume(torch.empty(0).to(device), D-1, None, None,torch.tensor(z[ind]).to(device) )


    # Volume FSC
    vol_fname = "data/cryofold/cryobench_IgD/IgG-1D/vols/128_org/%s.mrc"%str(iii).zfill(3)
    v = v.detach().cpu()
    # write_mrc("data/cryofold/cryobench_IgD/IgG-1D/images/snr0.01/test.mrc", v.numpy(), Apix=3.0)
    vol,header =parse_mrc(vol_fname)
    vol =  torch.tensor(vol)
    apix = header.apix
    fsc = get_fsc_curve(v, vol)
    fscauc = np.trapz(np.array(fsc["fsc"]), np.array(fsc["pixres"]))
    fsc_res = get_fsc_thresholds(fsc, apix=apix)
    # create_fsc_plot(fsc, "data/cryofold/cryobench_IgD/IgG-1D/images/snr0.01/run2/analysis/fsc.png", apix)

    # PDB RMSD
    pdb_fname =  "data/cryofold/cryobench_IgD/IgG-1D/pdbs/%s.pdb"%str(iii).zfill(3)
    tmp_fname =  "data/cryofold/cryobench_IgD/IgG-1D/images/snr0.01/test.pdb"
    s["final_atom_positions"] = s["final_atom_positions"] @ model.model.decoder.rot_init + model.model.decoder.trans_init + torch.tensor([64*3.0]).to(device)
    struct_to_pdb(tensor_tree_map(lambda x: x.detach().cpu().numpy()[-1], s),tmp_fname)
    parser = PDBParser(QUIET=True)
    structure1 = parser.get_structure("ref", tmp_fname)
    structure2 = parser.get_structure("mobile",pdb_fname)
    rmsd,_,_ = rmsd_best_permutation(structure1, structure2)
    print("RMSD : %.2f A"%(rmsd))


    outputs["fsc"].append(fsc)
    outputs["fscauc"].append(fscauc)
    outputs["fscres"].append(fsc_res)
    outputs["rmsd"].append(rmsd)

    with open("data/cryofold/cryobench_IgD/IgG-1D/images/snr0.01/run2/postprocess.pkl", "wb") as f:
        pickle.dump(outputs, f)

    del v, s
    import gc
    gc.collect()
    torch.cuda.empty_cache()

from matplotlib.ticker import FuncFormatter
plt.rcParams.update({'font.size': 18}) 
fig, ax = plt.subplots(1,1, figsize=(10,6), layout="constrained")
fsc_arr = np.array([o["fsc"].to_numpy() for o in outputs["fsc"]])
fsc_avg = fsc_arr.mean(axis=0)
fsc_std = fsc_arr.std(axis=0)
fsc_step = outputs["fsc"][0]["pixres"].to_numpy()
ax.errorbar(x=(fsc_step/apix), y=fsc_avg, yerr=fsc_std)
def fraction_formatter(x, pos):
    if x == 0:
        return "0"
    return f"1/{x**-1:.1f}"   # reciprocal with 2 decimal places

ax.xaxis.set_major_formatter(FuncFormatter(fraction_formatter))
ax.axhline(0.143, c="red")
ax.set_xlabel("Resolution ($1/\AA$)")
ax.set_ylabel("Fourier Shell Correlation")
fig.savefig("/home/vuillemr/cryofold/cryobench_IgD/IgG-1D/images/snr0.01/run2/analysis/test_fsc_avg.png", dpi=300)

print(np.mean(outputs["rmsd"]))
print(np.median(outputs["rmsd"]))
fig, ax = plt.subplots(1,1, figsize=(10,6), layout="constrained")
ax.hist(outputs["rmsd"], 100)
ax.set_xlabel("RMSD ($\AA$)")
ax.set_ylabel("")
ax.set_yscale("log")
fig.savefig("/home/vuillemr/cryofold/cryobench_IgD/IgG-1D/images/snr0.01/run2/analysis/test_rmsd.png", dpi=300)

fig, ax = plt.subplots(1,1, figsize=(10,6), layout="constrained")
ax.plot(outputs["rmsd"],outputs["fscauc"], ".", alpha=0.15, markersize=1)
ax.set_xlabel("RMSD ($\AA$)")
ax.set_ylabel("FSC AUC")
fig.savefig("/home/vuillemr/cryofold/cryobench_IgD/IgG-1D/images/snr0.01/run2/analysis/test_rmsd_res.png", dpi=300)


z_mu_all = np.array(z_mu_all)[:,-1 ,:]

import matplotlib.pyplot as plt
kwargs = {
    "c":[i for i in range(len(z_mu_all))],
    "cmap":"jet",
    "alpha":0.5,
    "s":0.1
}
fig, ax = plt.subplots(zdim-1,zdim-1, figsize=(10,10), layout="constrained")
for i in range(zdim-1):
    for j in range(1, zdim):
        if not i>=j:
            ax[j-1,i].scatter(z_mu_all[:,i], z_mu_all[:,j], **kwargs)
            ax[j-1,i].set_xlabel("Z%i"%i, fontsize=15)
            ax[j-1,i].set_ylabel("Z%i"%j, fontsize=15)
        else:
            ax[j-1,i].set_visible(False)

fig.savefig("/home/vuillemr/cryofold/cryobench_IgD/IgG-1D/images/snr0.01/run2/analysis/test_zmu_all.png", dpi=300)

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
# from ortools.constraint_solver import pywrapcp, routing_enums_pb2
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

# def solve_tsp(points):
#     n = len(points)
#     dist_matrix = np.linalg.norm(points[:,None] - points[None,:], axis=2)
    
#     # Setup routing
#     manager = pywrapcp.RoutingIndexManager(n, 1, 0)
#     routing = pywrapcp.RoutingModel(manager)
    
#     def distance_callback(i, j):
#         return int(dist_matrix[manager.IndexToNode(i)][manager.IndexToNode(j)] * 1000)
    
#     transit_callback_index = routing.RegisterTransitCallback(distance_callback)
#     routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
#     # Search
#     search_params = pywrapcp.DefaultRoutingSearchParameters()
#     search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
#     search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
#     search_params.time_limit.seconds = 10
    
#     solution = routing.SolveWithParameters(search_params)
    
#     # Extract route
#     route = []
#     idx = routing.Start(0)
#     while not routing.IsEnd(idx):
#         route.append(manager.IndexToNode(idx))
#         idx = solution.Value(routing.NextVar(idx))
#     return route


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

import torch 
from flexfold.models import HetOnlyVAE
from cryodrgn import config
from cryodrgn.utils import load_pkl
import matplotlib.pyplot as plt
z = load_pkl("/home/vuillemr/cryofold/cryobench_IgD/IgG-1D/images/snr0.01/run2/z.20.pkl")
zdim = z.shape[-1]

dimred = UMAP(n_components=2, n_neighbors=50, min_dist=0.1, random_state=42)
# dimred = PCA(n_components=2)
data = dimred.fit_transform(z)

n_clusters=8
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



traj_center = centers[ordered_idx]
traj_center = np.concatenate((traj_center, traj_center[[0]]))
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from skimage import measure
from skimage.morphology import medial_axis
from scipy.ndimage import laplace
from scipy.ndimage import gaussian_filter

points = data.T
x = data[:,0]
y = data[:,1]
# KDE on grid
kde = gaussian_kde(points)
xmin, xmax = x.min(), x.max()
ymin, ymax = y.min(), y.max()
npix = 200
X, Y = np.mgrid[xmin:xmax:npix*1j, ymin:ymax:npix*1j]
Z = kde(np.vstack([X.ravel(), Y.ravel()]))

# Find contour of high-density region (e.g. 70% iso-probability line)
Z = Z.reshape(X.shape)
mask = Z > Z.max()/2
skeleton, distance = medial_axis(mask, return_distance=True)
Z_g = gaussian_filter(skeleton.astype(float), sigma=5)
# contours = measure.find_contours(Z_g, np.percentile(Z_g, 99))
xx,yy = np.nonzero(skeleton)
ind = tsp_bidirectional(np.array([xx, yy]).T)
xx = xx[ind]
yy = yy[ind]
xx= ((xx)/(npix-1))* (xmax-xmin) + xmin
yy = ((yy)/(npix-1))* (ymax-ymin) + ymin

fig, ax = plt.subplots(1,1)
ax.scatter(x, y, s=1, alpha=0.1, c="orange",)
ax.plot(xx, yy, "-", c="r", markersize=1)
ax.plot(x.reshape(50,z.shape[0]//50).mean(axis=1), y.reshape(50,z.shape[0]//50).mean(axis=1), "o", c="orange", markersize=10)
ax.imshow(Z.T, extent=[xmin, xmax, ymax,ymin], cmap="Blues")
fig.savefig("/home/vuillemr/cryofold/cryobench_IgD/IgG-1D/images/snr0.01/run2/analysis/test_umap_smooth.png", dpi=300)


# z_traj = dimred.inverse_transform(np.array([xx,yy]).T)
# z_traj_downsampled = z_traj[::10]

traj = np.array([xx,yy])
ind = np.argmin(np.linalg.norm(traj[:, None, : ] - points[:,  :, None ], axis=0), axis=0)
z_traj = z[ind]
z_traj_downsampled = z_traj[::9][:51]


z_traj_downsampled = z.reshape(50,z.shape[0]//50, z.shape[-1]).mean(axis=1)


##################################################""
# output models ztraj
##################################################""
from flexfold.core import struct_to_pdb
from openfold.utils.tensor_utils import tensor_tree_map
from cryodrgn.mrcfile import write_mrc

model.to(device)

model.decoder.all_atom =True
model.decoder.pixel_size = 1.5
model.decoder.sigma = 1.05

for i in range(z_traj_downsampled.shape[0]):
    print(i)
    prefix = "/home/vuillemr/cryofold/cryobench_IgD/IgG-1D/images/snr0.01/run2/analysis/pc_traj/%s"%str(i+1).zfill(5)
    vol, struct = model.decoder.eval_volume(lattice.coords, D, lattice.extent,norm, z_traj_downsampled[i])
    crd = struct["final_atom_positions"]
    crd = crd @ model.decoder.rot_init + model.decoder.trans_init[..., None, :]
    struct["final_atom_positions"] = crd
    struct_to_pdb(tensor_tree_map(lambda x: x.detach().cpu().numpy()[-1], struct), 
                  prefix+".pdb")

    write_mrc(prefix+".mrc", np.array(vol.cpu()).astype(np.float32), Apix=model.decoder.pixel_size)

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


#
#########CRYODRGN###############################################################""
# CRYODRGN
# CRYODRGN
# CRYODRGN
# CRYODRGN
# CRYODRGN
# CRYODRGN
#########CRYODRGN###############################################################""
import torch 
from cryodrgn.models import HetOnlyVAE
from cryodrgn import config
from cryodrgn.utils import load_pkl
import matplotlib.pyplot as plt

config_file = "/home/vuillemr/cryofold/cryobench_IgD/IgG-1D/images/snr0.01/run_cryodrgn/config.yaml"
weight_file = "/home/vuillemr/cryofold/cryobench_IgD/IgG-1D/images/snr0.01/run_cryodrgn/weights.99.pkl"

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
z = load_pkl("/home/vuillemr/cryofold/cryobench_IgD/IgG-1D/images/snr0.01/run_cryodrgn/z.99.pkl")
zdim = z.shape[-1]


dimred = UMAP(n_components=2, random_state=42)
# dimred = PCA(n_components=2)
data = dimred.fit_transform(z)

n_clusters=8
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
ax.scatter(data[:,0], data[:,1], cmap="hsv", alpha=0.5, c=np.arange(100000), s =10)
ax.plot(traj_center[:,0], traj_center[:,1], "-", c="black")
# ax.scatter(traj_center[:,0], traj_center[:,1],s=200, cmap=cmap, c=[i for i in range(n_clusters)], edgecolors="black")
fig.savefig("/home/vuillemr/cryofold/cryobench_IgD/IgG-1D/images/snr0.01/run_cryodrgn/analyze.100/test_umap.png")



traj_center = centers[ordered_idx]
traj_center = np.concatenate((traj_center, traj_center[[0]]))
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from skimage import measure
from skimage.morphology import medial_axis
from scipy.ndimage import laplace
from scipy.ndimage import gaussian_filter

points = data.T
x = data[:,0]
y = data[:,1]
# KDE on grid
kde = gaussian_kde(points)
xmin, xmax = x.min(), x.max()
ymin, ymax = y.min(), y.max()
npix = 100
X, Y = np.mgrid[xmin:xmax:npix*1j, ymin:ymax:npix*1j]
Z = kde(np.vstack([X.ravel(), Y.ravel()]))

# Find contour of high-density region (e.g. 70% iso-probability line)
Z = Z.reshape(X.shape)
mask = Z > Z.max()/2
skeleton, distance = medial_axis(mask, return_distance=True)
Z_g = gaussian_filter(skeleton.astype(float), sigma=5)
# contours = measure.find_contours(Z_g, np.percentile(Z_g, 99))
xx,yy = np.nonzero(skeleton)
ind = tsp_bidirectional(np.array([xx, yy]).T)
xx = xx[ind]
yy = yy[ind]
xx= ((xx)/(npix-1))* (xmax-xmin) + xmin
yy = ((yy)/(npix-1))* (ymax-ymin) + ymin

fig, ax = plt.subplots(1,1)
ax.scatter(x, y, s=1, alpha=0.1, c="orange",)
ax.plot(xx, yy, "-", c="r", markersize=1)
ax.imshow(Z.T, extent=[xmin, xmax, ymax,ymin], cmap="Blues")
fig.savefig("/home/vuillemr/cryofold/cryobench_IgD/IgG-1D/images/snr0.01/run_cryodrgn/analyze.100/test_umap_smooth.png", dpi=300)


traj = np.array([xx,yy])
ind = np.argmin(np.linalg.norm(traj[:, None, : ] - points[:,  :, None ], axis=0), axis=0)
z_traj = z[ind]
z_traj_downsampled = z_traj[::5][:51]

z_traj_downsampled = z.reshape(50,z.shape[0]//50,z.shape[-1]).mean(axis=1)

model.to(device)

for i in range(z_traj_downsampled.shape[0]):
    print(i)
    prefix = "/home/vuillemr/cryofold/cryobench_IgD/IgG-1D/images/snr0.01/run_cryodrgn/analyze.100/pc_traj/%s"%str(i+1).zfill(5)
    vol = model.decoder.eval_volume(lattice.coords, D, lattice.extent,norm, z_traj_downsampled[i])
    write_mrc(prefix+".mrc", np.array(vol.cpu()).astype(np.float32), Apix=3.0)
